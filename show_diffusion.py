import argparse
import os
import os.path as osp
import shutil
import time
import warnings
from mmcv.image import tensor2imgs
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes
from mmseg.ops import resize
import torch.nn.functional as F
import cv2
import numpy as np
from torch import multiprocessing as mp
import einops
config="configs/segdiffusion/upernet_beit-base_512x512_160k_ade20k_20t_kl_loss.py"
checkpoint="work_dirs/upernet_beit-base_512x512_160k_ade20k_20t_kl_loss/iter_160000.pth"
out_dir="work_dirs/upernet_beit-base_512x512_160k_ade20k_20t_kl_loss/visualize_data"
base="base_512x512_160k_ade20k_20t_kl_loss"
inference_with_uc=True

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

# def mod_log_z_by_uc(log_z,uc,t):
#     rate=1.0/20.0/19.0*t
#     gate=torch.quantile(uc, 1-rate)
#     to_mask=uc>gate
#     mask=torch.zeros_like(log_z)
#     mask[:,-1]=1
#     mask = torch.log(mask.clamp(min=1e-30))
#     p=to_mask*mask+(~to_mask)*log_z
#     log_z.data.copy_(p)

def mod_log_z_by_uc(log_z,uc,t,log_cumprod_ct,x_recon):
    ctt=torch.exp(log_cumprod_ct[t-1]).item()
    gate=torch.quantile(uc, ctt)
    to_mask=uc>gate
    mask=torch.zeros_like(log_z)
    mask[:,-1]=1
    mask = torch.log(mask.clamp(min=1e-30))
    p=to_mask*mask+(~to_mask)*index_to_log_onehot(x_recon,log_z.shape[1])
    log_z.data.copy_(p)

# def extract_uc(logits_aux):
#     #print(logits_aux.sum(dim=1))
#     x=einops.repeat(-torch.log(logits_aux),"B C H W -> B C 9 H W")
#     B,C,_,H,W=x.shape
#     logits_aux=F.pad(logits_aux,(1,1,1,1),"reflect")
#     logits_aux=F.unfold(logits_aux,3,stride=1)
#     logits_aux=einops.rearrange(logits_aux,"B (c t1 t2) (H W) -> B c (t1 t2) H W",t1=3,t2=3,H=H,W=W)
#     uc_map=torch.einsum("abcde,abcde->ade",x,logits_aux)/9
#     return uc_map.unsqueeze(1)

def extract_uc(logits_aux):
    uc_map=1-torch.max(logits_aux,dim=1)[0]
    return uc_map.unsqueeze(1)

def worker(rank,config,checkpoint,out_dir,world_size):
    torch.cuda.set_device(rank)
    cfg = mmcv.Config.fromfile(config)
    dataset = build_dataset(cfg.data.test)
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1,
        dist=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE
    torch.cuda.empty_cache()
    model = revert_sync_batchnorm(model)
    #model = MMDataParallel(model, device_ids=gpu_ids)
    model=model.cuda()
    single_gpu_test(model,data_loader,out_dir,0.5,world_size,rank)

def make_grid(imgs,rows=0,margin=5):
    H,W,C=imgs[0].shape
    li=len(imgs)
    if not rows:
        rows=int(np.floor(np.sqrt(li)))
    cols=int(np.ceil(li/rows))
    pad=np.zeros(((H+margin)*rows-margin,(W+margin)*cols-margin,3),dtype=np.uint8)
    for  i in range(li):
        rs=i//cols
        ls=i%cols
        pad[rs*(H+margin):rs*(H+margin)+H,ls*(W+margin):ls*(W+margin)+W]=imgs[i]
    return pad

def vis_uc(x,color):
    color=np.array(color).reshape(1,1,3)
    x=x/np.max(x)
    return np.expand_dims(x,-1)*color
    

def slide_inference(model, img):
    h_stride, w_stride = model.test_cfg.stride
    h_crop, w_crop = model.test_cfg.crop_size
    batch_size, _, h_img, w_img = img.size()
    num_classes = model.num_classes
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    crop_images=[]
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_images.append(crop_img)
    def merge_fn(patches):
        preds = torch.zeros((batch_size, num_classes+1, h_img, w_img),device=patches[0].device)
        count_mat = torch.zeros((batch_size, 1, h_img, w_img),device=patches[0].device)
        i=0
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_seg_logit = patches[i]
                B,C,H,W=crop_seg_logit.shape
                i+=1
                preds[:,:C,y1:y2, x1:x2]+=crop_seg_logit
                # preds += F.pad(crop_seg_logit,
                #                 (int(x1), int(preds.shape[3] - x2), int(y1),
                #                 int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds
    return crop_images, merge_fn


def single_gpu_test(model,data_loader,out_dir,opacity=0.5,world_size=1,rank=0):
    model.eval()
    results = []
    dataset = data_loader.dataset
    if rank==0:
        prog_bar = mmcv.ProgressBar(len(dataset)//world_size)
    loader_indices = data_loader.batch_sampler
    counter=0
    os.makedirs(out_dir,exist_ok=True)
    for batch_indices, data in zip(loader_indices, data_loader):
        if counter%world_size==rank:
            to_save=dict()
            with torch.no_grad():
                img_tensor = data['img'][0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                gt=dataset.get_gt_seg_map_by_idx(counter)
                gt=torch.from_numpy(gt).unsqueeze(0).to(dtype=torch.uint8)-1
                assert len(imgs) == len(img_metas)
                crop_images, merge_fn = slide_inference(model, img_tensor.cuda())
                ts=[[] for i in range(len(crop_images))]
                zs=[[] for i in range(len(crop_images))]
                xs=[[] for i in range(len(crop_images))]
                outs=[]
                temp_uc=[0]
                for i in range(len(crop_images)):
                    def call_back(log_z,log_x_recon,t,x_recon,**args):
                        ts[i].append(t[0].item())
                        xs[i].append(resize(input=torch.exp(log_x_recon),size=crop_images[0].shape[2:],mode='bilinear',align_corners=model.align_corners))
                        if inference_with_uc:
                            if ts[i][-1]==19:
                                temp_uc_=extract_uc(xs[i][-1])
                                temp_uc[0]=temp_uc_/temp_uc_.max()
                            mod_log_z_by_uc(log_z,temp_uc[0],ts[i][-1],model.log_cumprod_ct,x_recon)
                        # zs[i].append(resize(input=torch.exp(log_z),size=crop_images[0].shape[2:],mode='bilinear',align_corners=model.align_corners))
                        
                    out = model.sample(crop_images[i],return_logits = True,call_back=call_back)
                    out = out['logits']
                    out = resize(input=out,size=crop_images[0].shape[2:],mode='bilinear',align_corners=model.align_corners)
                    outs.append(out)
                out=merge_fn(outs)
                # zs=[merge_fn(i) for i in  zip(*zs)]
                xs=[merge_fn(i) for i in  zip(*xs)]
                ts=ts[0]
                img=imgs[0]
                img_meta=img_metas[0]
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                ori_h, ori_w = img_meta['ori_shape'][:-1]
                uc_map=[resize(extract_uc(i), (ori_h, ori_w),mode='bilinear',align_corners=model.align_corners) for i in xs]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                showed_gt=model.show_result(img_show,gt,opacity=opacity)
                xs=[torch.argmax(resize(i, (ori_h, ori_w),mode='bilinear',align_corners=model.align_corners),1).cpu() for i in xs]
                # zs=[torch.argmax(resize(i, (ori_h, ori_w),mode='bilinear',align_corners=model.align_corners),1).cpu() for i in zs]
                errs=[(i!=gt) & (gt !=255) for i in xs]
                uc_map=[(vis_uc(i.squeeze().cpu().numpy(),[255,255,255])).astype("uint8") for i in uc_map]
                uc_map=make_grid(uc_map)
                cv2.imwrite("%s/%s_uncertainty.png"%(out_dir,counter),uc_map)
                err_map=[(vis_uc(i.squeeze().cpu().numpy(),[255,255,255])).astype("uint8") for i in errs]
                err_map=make_grid(err_map)
                cv2.imwrite("%s/%s_err.png"%(out_dir,counter),err_map)
                ### seg
                seg=torch.argmax(resize(out, (ori_h, ori_w),mode='bilinear',align_corners=model.align_corners),1).cpu()
                showed_img_out=model.show_result(img_show,seg,opacity=opacity)
                cv2.imwrite("%s/%s_seg.png"%(out_dir,counter),showed_img_out)
                to_save['seg']=showed_img_out
                ### gt
                cv2.imwrite("%s/%s_seg_gt.png"%(out_dir,counter),showed_gt)
                ### zt
                # showed_zt_out=make_grid([model.show_result(img_show,i,opacity=opacity) for i in zs])
                # cv2.imwrite("%s/%s_zt.png"%(out_dir,counter),showed_zt_out)
                # to_save['zt']=showed_zt_out
                ### xt
                showed_xt_out=make_grid([model.show_result(img_show,i,opacity=opacity) for i in xs])
                cv2.imwrite("%s/%s_xt.png"%(out_dir,counter),showed_xt_out)
                to_save['xt']=xs
                torch.cuda.empty_cache() 
                torch.save(to_save,"%s/%s_data.pkl"%(out_dir,counter))
                if rank==0:
                    prog_bar.update()
        counter+=1
        

def eval(config,out_dir):
    cfg = mmcv.Config.fromfile(config)
    dataset = build_dataset(cfg.data.test)
    import glob
    import json
    files=glob.glob("%s/*_data.pkl"%out_dir)
    files=sorted(files,key=lambda x:int(x.split('/')[-1][:-9]))
    xts=[torch.load(i)['xt'] for i in files]
    res=[]
    for T in range(len(xts[0])):
        g=[i[T].cpu().numpy().squeeze() for i in xts]
        miou = dataset.evaluate(g, metric="mIoU")
        res.append(miou)
    with open("%s/metrics.json"%out_dir,"w") as f:
        json.dump(res,f)
    targets=['mIoU','aAcc','mAcc']
    r2=[]
    for i in range(len(res)):
        g={j:res[-1-i][j] for j in targets}
        g["t"]=i
        r2.append(g)
    print(r2)
    with open("%s/metrics_simple.json"%out_dir,"w") as f:
        json.dump(r2,f)

        

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default=config)
    parser.add_argument("--checkpoint",default=checkpoint)
    parser.add_argument("--out_dir",default=out_dir)
    # parser.add_argument("--base",default=base)
    args=parser.parse_args()
    mp.set_start_method('spawn',force=True)
    world_size=torch.cuda.device_count()
    mp.spawn(worker,nprocs=world_size,args=(args.config,args.checkpoint,args.out_dir,world_size),daemon=False)
    eval(config,out_dir)