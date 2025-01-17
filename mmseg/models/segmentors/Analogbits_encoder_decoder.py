# Copyright (c) OpenMMLab. All rights reserved.
from distutils.fancy_getopt import OptionDummy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from .Analogbits_gaussian_diffusion_utils import ABGaussianDiffusionSeg
from .Analogbits_Uperhead import Uper_decode_head
from .analogbits_Unet.unet import UNet as Analogbits_Unet
import einops

def num2nb(x,N):
    assert x<2**N
    bins=bin(x)[2:]
    bins='0'*(N-len(bins))+bins
    return [float(i) for i in bins]

def nb2num(L):
    c=0
    for i,j in enumerate(L[::-1]):
        c+=j*2**i
    return c 

@SEGMENTORS.register_module()
class AnalogBitsEncoderDecoder(BaseSegmentor,ABGaussianDiffusionSeg):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone=None,
                 mask_Unet=None,
                 decode_head=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 diffusion_cfg=None,
                 **kwargs):
        BaseSegmentor.__init__(self,init_cfg)
        self.num_classes=auxiliary_head['num_classes']
        self.num_bits=int(np.ceil(np.log2(self.num_classes)))
        self.register_buffer('bit_mask',torch.tensor([2**i for i in range(self.num_bits)][::-1]))
        self.register_buffer('bit_emb',torch.tensor([ num2nb(i,self.num_bits) for i in range(self.num_classes)]))
        ABGaussianDiffusionSeg.__init__(self,**diffusion_cfg)
        self.backbone = builder.build_backbone(backbone)
        self._init_mask_Unet(mask_Unet,diffusion_cfg)
        self._init_auxiliary_head(auxiliary_head)
        decode_head['out_channels']=[i*mask_Unet['inner_channel'] for i in mask_Unet['channel_mults'][1:5]]
        self._init_decode_head(decode_head)
        self.neck=builder.build_neck(dict(type='Feature2Pyramid',embed_dim=decode_head['channels'],rescales=[4, 2, 1, 0.5],norm_cfg=dict(type='SyncBN', requires_grad=True)))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_cache=False
        self.cache=None
        #assert self.with_decode_head

    def mask_to_bits(self,mask):
        bits=F.embedding(mask,self.bit_emb)
        return einops.rearrange(bits,'b h w c -> b c h w')
    
    def bits_to_mask(self,bits):
        bits=torch.round(bits)
        mask=torch.einsum('bchw,c->bhw',bits,self.bit_mask).long()
        return mask

    def quantize_logits(self,bits):
        dist=torch.sum((bits.unsqueeze(1)-einops.rearrange(self.bit_emb,'n k -> 1 n k 1 1'))**2,dim=2) # b n h w
        return -dist

    def _init_mask_Unet(self,mask_Unet,diffusion_cfg):
        """the model encodes mask x_(t-1) to mask features, the model depends on t"""
        #mask_Unet['diffusion_step']=diffusion_cfg['diffusion_step']
        mask_Unet['in_channel']=self.num_bits if not diffusion_cfg['use_self_condition'] else self.num_bits*2
        mask_Unet['out_channel']=self.num_bits
        self.mask_Unet=Analogbits_Unet(**mask_Unet)

    def _init_auxiliary_head(self, auxiliary_head):
        """the uxiliary head use only image features for predict mask"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def _init_decode_head(self, decode_head):
        """the decode head use mixed mask and image features for predict mask, the model doesn't depend on t, so we can use mmseg models"""
        self.decode_head = Uper_decode_head(**decode_head)
        self.align_corners = self.decode_head.align_corners

    def _model(self ,x_t, t, **model_kwargs):
        im = model_kwargs['image']
        self_cond_prev=model_kwargs.get('self_cond_prev',None)
        if self.use_cache:
            if self.cache is not None:
               image_features=self.cache
            else:
                self._backbone_feature=self.neck(self.backbone(im))
                image_features=self.decode_head(self._backbone_feature)
                self.cache=image_features
        else:
            image_features=self.decode_head(self.neck(self.backbone(im)))
        # mask_features=self.mask_backbone(x_t,t)
        # mixed_features=self.feature_mixer(image_features,mask_features,t)
        # output=self.decode_head(mixed_features)
        output=self.mask_Unet(x_t,image_features,self_cond_prev,t)
        if output.shape[2:]!=im.shape[2:]:
            output = resize(
                input=output,
                size=im.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return output

    def del_cache(self):
        self.cache=None
        self.use_cache=False

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        B,C,H,W=img.shape
        # out = self.p_sample_loop(
        # self._model,
        # [B,self.num_classes,H,W],
        # noise=None,
        # clip_denoised=True,
        # denoised_fn=None,
        # cond_fn=None,
        # model_kwargs={"image":img},
        # device=None,
        # progress=False,
        # call_back=None,
        # start_step=None
        # )
        out = self.ddim_sample_loop(
            self._model,
            [B,self.num_bits,H,W],
            noise=None,
            clip_denoised=True,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs={"image":img},
            device=None,
            progress=False,
            eta=0.0,
        )
        return self.quantize_logits(out)


    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses


    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        self.use_cache=True
        losses = dict()
        gt_bits=self.mask_to_bits(torch.clamp(gt_semantic_seg,0,self.num_classes-1).squeeze(1))
        loss_decode = self.train_loss(batch=dict(image=img,seg=gt_bits),return_loss=True,gt_seg=gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(self._backbone_feature, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        self.del_cache()
        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
