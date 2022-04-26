# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import build_norm_layer
import torch
import math
from ..builder import NECKS

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
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.emb_type=emb_type
        if 'layernorm' in self.emb_type:
            self.norm = nn.LayerNorm(n_embd, elementwise_affine=False)
        if 'batchnorm' in self.emb_type:
            self.norm = nn.SyncBatchNorm(n_embd,affine=False)


    def forward(self, x, timestep):
        emb = self.linear(self.emb(timestep))
        if len(emb.shape)==3:
            emb=emb.unsqueeze(1)
            scale, shift = torch.chunk(emb, 2, dim=2)
        if len(emb.shape)==4:
            emb=emb.unsqueeze(-1).unsqueeze(-1)
            scale, shift = torch.chunk(emb, 2, dim=1)
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
                 diffusion_step,
                 rescales=[4, 2, 1, 0.5],
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.image_convs=nn.ModuleList([nn.Conv2d(image_feature_dim,embed_dim,3,padding=1) for i in range(4)])
        self.mask_convs=nn.ModuleList([nn.Conv2d(mask_feature_dim,embed_dim,3,padding=1) for i in range(4)])
        self.image_adanorms=nn.ModuleList([AdaNorm(embed_dim,diffusion_step,"adabatchnorm_abs") for i in range(4)])
        self.mask_adanorms=nn.ModuleList([AdaNorm(embed_dim,diffusion_step,"adabatchnorm_abs") for i in range(4)])
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                    build_norm_layer(norm_cfg, embed_dim)[1],
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 2:
                self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2))
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')

    def forward(self, image_features, mask_features, t):
        assert len(image_features) == len(self.rescales)
        outputs = []
        if self.upsample_4x is not None:
            ops = [
                self.upsample_4x, self.upsample_2x, self.identity,
                self.downsample_2x
            ]
        else:
            ops = [
                self.upsample_2x, self.identity, self.downsample_2x,
                self.downsample_4x
            ]
        for i in range(len(image_features)):
            outputs.append(ops[i](self.image_adanorms[i](self.image_convs[i](image_features[i]),t)+self.mask_adanorms[i](self.mask_convs[i](mask_features[i]),t)))
        return tuple(outputs)


@NECKS.register_module()
class MixCrossAttn(nn.Module):
    def __init__(self,
                 image_feature_dim,
                 mask_feature_dim,
                 embed_dim,
                 diffusion_step,
                 rescales=[4, 2, 1, 0.5],
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.image_convs=nn.ModuleList([nn.Conv2d(image_feature_dim,embed_dim,3,padding=1) for i in range(4)])
        self.mask_convs=nn.ModuleList([nn.Conv2d(mask_feature_dim,embed_dim,3,padding=1) for i in range(4)])
        self.image_adanorms=nn.ModuleList([AdaNorm(embed_dim,diffusion_step,"adabatchnorm_abs") for i in range(4)])
        self.mask_adanorms=nn.ModuleList([AdaNorm(embed_dim,diffusion_step,"adabatchnorm_abs") for i in range(4)])
        self.rescales = rescales
        self.upsample_4x = None
        for k in self.rescales:
            if k == 4:
                self.upsample_4x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                    build_norm_layer(norm_cfg, embed_dim)[1],
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2),
                )
            elif k == 2:
                self.upsample_2x = nn.Sequential(
                    nn.ConvTranspose2d(
                        embed_dim, embed_dim, kernel_size=2, stride=2))
            elif k == 1:
                self.identity = nn.Identity()
            elif k == 0.5:
                self.downsample_2x = nn.MaxPool2d(kernel_size=2, stride=2)
            elif k == 0.25:
                self.downsample_4x = nn.MaxPool2d(kernel_size=4, stride=4)
            else:
                raise KeyError(f'invalid {k} for feature2pyramid')

    def forward(self, image_features, mask_features, t):
        assert len(image_features) == len(self.rescales)
        outputs = []
        if self.upsample_4x is not None:
            ops = [
                self.upsample_4x, self.upsample_2x, self.identity,
                self.downsample_2x
            ]
        else:
            ops = [
                self.upsample_2x, self.identity, self.downsample_2x,
                self.downsample_4x
            ]
        for i in range(len(image_features)):
            outputs.append(ops[i](self.image_adanorms[i](self.image_convs[i](image_features[i]),t)+self.mask_adanorms[i](self.mask_convs[i](mask_features[i]),t)))
        return tuple(outputs)
