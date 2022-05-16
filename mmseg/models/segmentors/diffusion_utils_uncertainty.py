from sched import scheduler
from time import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import einops
from abc import ABC, abstractmethod
torch.autograd.set_detect_anomaly(True)
eps = 1e-8

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def _extract(x,t,x_shape):
    b, *_ = t.shape
    out = x.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DiffusionSegUC(ABC):
    def __init__(
        self,
        *,
        num_classes=150,
        diffusion_step=20,
        alpha_init_type='alpha1',
        loss_weights=[1,1],
        adaptive_auxiliary_loss=False,
        ignore_class=255,
        t_sampler="importance",
        schedule_params=dict(ctt_1=0.05, ctt_T=0.99999, snr_1=0.9, snr_T=0.05,cttr=0.5)
    ):
        super().__init__()

        self.loss_weights=loss_weights
        self.adaptive_auxiliary_loss=adaptive_auxiliary_loss
        self.num_classes = num_classes
        self.amp=False
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'
        self.ignore_class=ignore_class
        self.t_sampler=t_sampler
        self.alpha_init_type=alpha_init_type
        self.alpha_schedule(alpha_init_type,schedule_params)
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))
        self.zero_vector = None

    def alpha_schedule(self,alpha_init_type,schedule_params):
        assert alpha_init_type in ["alpha1"]
        if alpha_init_type=="alpha1":
            ctt_1=schedule_params['ctt_1']
            ctt_T=schedule_params['ctt_T']
            snr_1=schedule_params['snr_1']
            snr_T=schedule_params['snr_T']
            self.cttr=schedule_params['cttr']
            N=self.num_timesteps
            time_step=self.num_timesteps
            ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
            ctt = np.concatenate(([0], ctt))
            ctt = torch.tensor(ctt.astype('float64'))
            log_cumprod_ct = torch.log(ctt)
            self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
            snr = np.arange(0, time_step)/(time_step-1)*(snr_T - snr_1) + snr_1
            snr = np.concatenate(([1], snr))
            snr = torch.tensor(snr)
            self.register_buffer('snr', snr)
            
    def modify_ctt_by_uc(self,uc,log_ctt):
        return log_ctt*(1-self.cttr*uc)

    def modify_snr_by_uc(self,uc,snr):
        pass

    def extract(self, t, x_shape,uc_map):
        #### modify current noise scheduler by uncertainty.
        device=uc_map.device
        scheduler_args=dict()
        with torch.no_grad():
            log_ctt=self.modify_ctt_by_uc(uc_map,_extract(self.log_cumprod_ct,t,x_shape))
            log_ctt_next=self.modify_ctt_by_uc(uc_map,_extract(self.log_cumprod_ct,t+1,x_shape))
            log_ct=torch.log(1-(1-torch.exp(log_ctt_next))/(1-torch.exp(log_ctt)))
            snr=_extract(self.snr,t,x_shape)
            snr_next=_extract(self.snr,t+1,x_shape)
            log_1_min_cumprod_ct=log_1_min_a(log_ctt)
            log_1_min_ct=log_1_min_a(log_ct)
            log_att=log_1_min_cumprod_ct+torch.log(snr)
            log_1_min_cumprod_ct_next= log_1_min_a(log_ctt_next)
            log_att_next= log_1_min_cumprod_ct_next+torch.log(snr_next)
            log_btt_next=log_att_next+torch.log((1-snr_next)/self.num_classes/snr_next)
            log_at=log_att_next-log_att
            scheduler_args['log_cumprod_at']=log_att_next
            scheduler_args['log_cumprod_bt']=log_btt_next
            scheduler_args['log_cumprod_ct']=log_ctt_next
            scheduler_args['log_at']=log_at
            scheduler_args['log_bt']=torch.log((1-torch.exp(log_at)-torch.exp(log_ct))/self.num_classes)
            scheduler_args['log_ct']=log_ct
            scheduler_args['log_1_min_ct'] = log_1_min_ct
            scheduler_args['log_1_min_cumprod_ct'] = log_1_min_cumprod_ct_next
        return scheduler_args

    @abstractmethod
    def _model(im ,x_t, t, uc_map):
        pass

    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t,scheduler_args):         # q(xt|xt_1)
        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:,:-1,:]+scheduler_args['log_at'], scheduler_args['log_bt']),
                log_add_exp(log_x_t[:, -1:, :] + scheduler_args['log_1_min_ct'], scheduler_args['log_ct'])
            ],
            dim=1
        )

        return log_probs

    def q_pred(self, log_x_start, scheduler_args):           # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:,:-1,:]+scheduler_args['log_cumprod_at'], scheduler_args['log_cumprod_bt']),
                log_add_exp(log_x_start[:,-1:,:]+scheduler_args['log_1_min_cumprod_ct'], scheduler_args['log_cumprod_ct'])
            ],
            dim=1
        )

        return log_probs

    def predict_start(self, log_x_t, im, t, uc_map):          # p(x0|xt)
        x_t = torch.exp(log_x_t)
        if self.amp == True:
            with autocast():
                out = self._model(im ,x_t, t, uc_map)
        else:
            out = self._model(im, x_t, t, uc_map)
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes
        assert out.size()[2:] == x_t.size()[2:]
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        #if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
        self.zero_vector = torch.zeros(batch_size, 1,log_pred.shape[2],log_pred.shape[3]).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)

        return log_pred

    def q_posterior(self, log_x_start, log_x_t, scheduler_args, scheduler_args_1):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes).unsqueeze(1) 
        log_one_vector = torch.zeros(batch_size,1,1,1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, log_x_start.shape[2],log_x_start.shape[3])

        log_qt = self.q_pred(log_x_t, scheduler_args)                                  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = scheduler_args['log_cumprod_ct']
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes, -1,-1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, scheduler_args)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        ct_vector = scheduler_args['log_ct'].expand(-1, self.num_classes, -1,-1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        
        # log_x_start = torch.cat((log_x_start, log_zero_vector), dim=1)
        # q = log_x_start - log_qt
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, scheduler_args_1) + log_qt_one_timestep + q_log_sum_exp
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)


    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes+1)
        return log_sample

    def q_sample(self, log_x_start, scheduler_args):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, scheduler_args)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        elif type(method)==list:
            t=np.random.choice(method,b)
            t=torch.from_numpy(t).to(device=device,dtype=torch.int64)
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _train_loss(self, x, im, is_train=True,t=None,uc_map=None):                       # get the KL loss
        b, device = x.size(0), x.device
        x_start = x
        #### convert ignore class 255 to mask class 150
        x=x.clip(0,self.num_classes)
        if t is not None:
            t=torch.ones(b,dtype=torch.long,device=device)*t
            pt=torch.ones(b,device=device)
        else:
            t, pt = self.sample_time(b, device, self.t_sampler)
        log_x_start = index_to_log_onehot(x, self.num_classes+1)
        H,W=x.shape[1:]
        log_x_start_blur=einops.rearrange(F.one_hot(x,self.num_classes+1),"B H W C -> B C H W").float()
        log_x_start_blur=F.interpolate(log_x_start_blur,(H//4,W//4),mode='bilinear',align_corners=self.align_corners)
        log_x_start_blur=F.interpolate(log_x_start_blur,(H,W),mode='bilinear',align_corners=self.align_corners)
        log_x_start_blur = torch.log(log_x_start_blur.clamp(min=1e-30))
        scheduler_args=self.extract(t,log_x_start.shape,uc_map)
        scheduler_args_1=self.extract(t-1,log_x_start.shape,uc_map)
        log_xt = self.q_sample(log_x_start=log_x_start_blur, scheduler_args=scheduler_args)
        # log_xt.data.fill_(-30.0)
        # log_xt[:,150].data.fill_(0.0)
        # xt = log_onehot_to_index(log_xt)
        # num_unmask=(xt!=150).sum().item()
        # print(f"image size {xt.shape[1:]}")
        # print("umasked tokens %s"%num_unmask)
        # print("umask rate ",num_unmask/(xt.shape[2]*xt.shape[1]))

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, im, t=t, uc_map=uc_map)            # P_theta(x0|xt)
        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, scheduler_args=scheduler_args, scheduler_args_1=scheduler_args_1)      # go through q(xt_1|xt,x0)

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start
        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)
        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt,scheduler_args=scheduler_args, scheduler_args_1=scheduler_args_1)
        kl = self.multinomial_kl(log_true_prob, log_model_prob)  ## kl (B,H,W)
        #mask_region = (xt == self.num_classes).float()
        mask_region = (x_start == self.ignore_class).float()
        kl = kl * (1-mask_region)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_x0_recon)
        #print("decoder nll ",decoder_nll.mean().item())
        decoder_nll = sum_except_batch(decoder_nll*(1-mask_region))

        mask = (t == torch.zeros_like(t)).float()
        kl_loss = mask * decoder_nll + (1. - mask) * kl
        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt 
        loss2 = 0
        if self.loss_weights[1] != 0 and is_train==True:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            kl_aux = kl_aux * (1. - mask_region) 
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1-t/self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * kl_aux_loss / pt
        vb_loss = self.loss_weights[0]*loss1+ self.loss_weights[1]*loss2
        sums=(sum_except_batch((1-mask_region))+1e-8)
        vb_loss=vb_loss/sums
        Lt2 = vb_loss.pow(2)
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))
        acc_seg=sum_except_batch((torch.argmax(log_x0_recon,dim=1)==x_start).float())/sums
        return log_model_prob, vb_loss, acc_seg


    def device(self):
        return self.log_at.device

    def train_loss(
            self, 
            batch=None, 
            return_loss=True, 
            return_logits=False, 
            t=None,
            uc_map=None,
            **kwargs):
        if kwargs.get('autocast') == True:
            self.amp = True
        batch_size = batch['image'].shape[0]
        device = batch['image'].device
        log_model_prob, loss, acc_seg = self._train_loss(batch['seg'], batch['image'],t=t,uc_map=uc_map)
        out = {}
        if return_logits:
            out['logits'] = torch.exp(log_model_prob)

        if return_loss:
            out['loss'] = loss.mean()
            out['acc_seg']=acc_seg.mean()
        self.amp = False
        return out


    @abstractmethod
    def del_cache(self):
        pass

    def sample(
            self,
            image,
            return_logits = False,
            skip_step = 0,
            downsample=1,
            call_back=None,
            uc_map=None,
            **kwargs):
        self.use_cache=True
        batch_size = image.shape[0] 
        device = self.log_at.device
        if self.log_cumprod_ct[-2]>-1: ## start with all mask !
            zero_logits = torch.zeros((batch_size, self.num_classes, image.shape[2]//downsample,image.shape[3]//downsample),device=device)
            one_logits = torch.ones((batch_size, 1, image.shape[2]//downsample,image.shape[3]//downsample),device=device)
        else: ## start with all random !
            randoms=torch.randint(0,self.num_classes,(batch_size,1,image.shape[2]//downsample,image.shape[3]//downsample),device=device)
            zero_logits = torch.zeros((batch_size, self.num_classes,image.shape[2]//downsample,image.shape[3]//downsample),device=device).scatter_(1,randoms,1)
            one_logits = torch.zeros((batch_size, 1,image.shape[2]//downsample,image.shape[3]//downsample),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        log_z = torch.log(mask_logits)
        start_step = self.num_timesteps
        with torch.no_grad():
            # skip_step = 1
            diffusion_list = [index for index in range(start_step-1, -1, -1-skip_step)]
            if diffusion_list[-1] != 0:
                diffusion_list.append(0)
            # for diffusion_index in range(start_step-1, -1, -1):
            for diffusion_index in diffusion_list:
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_x_recon = self.predict_start(log_z, image, t, uc_map)
                if diffusion_index > skip_step:
                    scheduler_args=self.extract(t-skip_step,log_z.shape,uc_map)
                    scheduler_args_1=self.extract(t-skip_step-1,log_z.shape,uc_map)
                    model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, scheduler_args=scheduler_args, scheduler_args_1=scheduler_args_1)
                else:
                    scheduler_args=self.extract(t,log_z.shape,uc_map)
                    scheduler_args_1=self.extract(t-1,log_z.shape,uc_map)
                    model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, scheduler_args=scheduler_args, scheduler_args_1=scheduler_args_1)

                log_z = self.log_sample_categorical(model_log_prob)
                if call_back:
                    call_back(log_z=log_z,log_x_recon=log_x_recon,t=t,x_recon=log_onehot_to_index(log_x_recon[:,:-1]))

        self.del_cache()
        content_token = log_onehot_to_index(log_x_recon)
        
        output = {'pred_seg': content_token}
        if return_logits:
            output['logits'] = torch.exp(log_z)
        return output
