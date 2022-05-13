# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_norm_layer
import torch
import math
from ..builder import NECKS
from .featurepyramid import Feature2Pyramid
class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        self.emb_type=emb_type
        super().__init__()
        affine=True
        if emb_type.startswith('ada') and diffusion_step>0:
            affine=False
            if "abs" in emb_type:
                self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
            else:
                self.emb = nn.Embedding(diffusion_step, n_embd)
            self.linear = nn.Linear(n_embd, n_embd*2)
        if 'layernorm' in self.emb_type:
            self.norm = nn.LayerNorm(n_embd, elementwise_affine=affine)
        if 'batchnorm' in self.emb_type:
            self.norm = nn.SyncBatchNorm(n_embd,affine=affine)


    def forward(self, x, timestep=None):
        if hasattr(self,'emb'):
            emb = self.linear(self.emb(timestep))
            if len(x.shape)==3:
                emb=emb.unsqueeze(1)
                scale, shift = torch.chunk(emb, 2, dim=2)
            if len(x.shape)==4:
                emb=emb.unsqueeze(-1).unsqueeze(-1)
                scale, shift = torch.chunk(emb, 2, dim=1)
        else:
            scale=0
            shift=0
        x = self.norm(x) * (1 + scale) + shift
        return x


@NECKS.register_module()
class MixerPyramid(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 image_feature_dim,
                 mask_feature_dim,
                 embed_dim,
                 diffusion_step=0,
                 rescales=[4, 2, 1, 0.5]):
        super().__init__()
        self.featurepyramid=Feature2Pyramid(embed_dim,rescales)
        self.image_convs=nn.ModuleList([nn.Conv2d(image_feature_dim,embed_dim,3,padding=1) for i in range(4)])
        self.mask_convs=nn.ModuleList([nn.Conv2d(mask_feature_dim,embed_dim,3,padding=1) for i in range(4)])
        self.image_adanorms=nn.ModuleList([AdaNorm(embed_dim,diffusion_step,"adabatchnorm_abs") for i in range(4)])
        self.mask_adanorms=nn.ModuleList([AdaNorm(embed_dim,diffusion_step,"adabatchnorm_abs") for i in range(4)])

    def forward(self, image_features, mask_features, t=None):
        outputs = []
        for i in range(len(image_features)):
            outputs.append(self.image_adanorms[i](self.image_convs[i](image_features[i]),t)+self.mask_adanorms[i](self.mask_convs[i](mask_features[i]),t))
        return tuple(self.featurepyramid(outputs))


@NECKS.register_module()
class MixerPyramid2(nn.Module):
    def __init__(self,
                 image_feature_dim,
                 mask_feature_dims,
                 embed_dim,
                 rescales=[4, 2, 1, 0.5]):
        super().__init__()
        self.featurepyramid=Feature2Pyramid(image_feature_dim,rescales)
        self.img_mask_convs=nn.ModuleList([nn.Conv2d(image_feature_dim+mask_feature_dims[i],embed_dim,1) for i in range(4)])


    def forward(self, image_features, mask_features, t=None):
        image_features=self.featurepyramid(image_features)
        outputs = []
        for i in range(len(image_features)):
            outputs.append(self.img_mask_convs[i](torch.cat([image_features[i],mask_features[i]],dim=1)))
        return tuple(outputs)


class MixerPyramidUC(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 image_feature_dim,
                 mask_feature_dim,
                 embed_dim,
                 diffusion_step=0,
                 rescales=[4, 2, 1, 0.5]):
        super().__init__()
        self.featurepyramid=Feature2Pyramid(embed_dim,rescales)
        self.image_convs=nn.ModuleList([nn.Conv2d(image_feature_dim,embed_dim,3,padding=1) for i in range(4)])
        self.mask_convs=nn.ModuleList([nn.Conv2d(mask_feature_dim,embed_dim,3,padding=1) for i in range(4)])
        self.image_adanorms=nn.ModuleList([AdaNorm(embed_dim,diffusion_step,"adabatchnorm_abs") for i in range(4)])
        self.mask_adanorms=nn.ModuleList([AdaNorm(embed_dim,diffusion_step,"adabatchnorm_abs") for i in range(4)])

    def forward(self, image_features, mask_features, t=None, uc_map=None):
        outputs = []
        for i in range(len(image_features)):
            outputs.append(self.image_adanorms[i](self.image_convs[i](image_features[i]),t)+(1-uc_map)*self.mask_adanorms[i](self.mask_convs[i](mask_features[i]),t))
        return tuple(self.featurepyramid(outputs))