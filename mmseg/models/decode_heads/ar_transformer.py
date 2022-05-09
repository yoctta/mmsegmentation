# Copyright (c) OpenMMLab. All rights reserved.
from turtle import forward
import einops
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmseg.utils import get_root_logger

# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numpy as np
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

def reduce_dict(ld):
    return {name:sum([i[name] for i in ld])/len(ld) for name in ld[0]}


class SelfAttention(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        if qv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.v_bias = None

        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] -
                                      1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        # coords shape is (2, Wh, Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        # coords_flatten shape is (2, Wh*Ww)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :])
        # relative_coords shape is (Wh*Ww, Wh*Ww, 2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1, ) * 2,
            dtype=relative_coords.dtype)
        # relative_position_index shape is (Wh*Ww, Wh*Ww)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer('relative_position_index',
                             relative_position_index)
        causal_mask=torch.triu(-float("inf")*torch.ones(relative_position_index.shape),1).unsqueeze(0)
        self.register_buffer("causal_mask",causal_mask)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x,context=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        if context is not None and len(context[0])>0:
            k=torch.cat([context[0],k],dim=-2)
            v=torch.cat([context[1],v],dim=-2)
            context=(k,v)
        attn = (q @ k.transpose(-2, -1))
        if self.relative_position_bias_table is not None:
            Wh = self.window_size[0]
            Ww = self.window_size[1]
            Laq,Lat=attn.shape[-2:]
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index[:Lat,:Lat].reshape(-1)].reshape(Lat,Lat, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            relative_position_bias=(self.causal_mask[:,:Lat,:Lat]+relative_position_bias)[:,-Laq:]
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, context


class CrossAttention(BaseModule):
    def __init__(self,
                 embed_dims,
                 embed_dims2,
                 num_heads,
                 window_size,
                 qv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        if qv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.v_bias = None

        self.window_size = window_size
        self.num_relative_distance = (2* window_size[0] -1) * (2* window_size[1] - 1) + 1
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))
        self.q_ = nn.Linear(embed_dims, embed_dims, bias=False)
        self.kv = nn.Linear(embed_dims2, embed_dims * 2, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        # coords shape is (2, Wh, Ww)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        # coords_flatten shape is (2, Wh*Ww)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :])
        # relative_coords shape is (Wh*Ww, Wh*Ww, 2)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # shift to start from 0
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1, window_size[0] * window_size[1] ),
            dtype=relative_coords.dtype)
        # relative_position_index shape is (Wh*Ww, Wh*Ww)
        relative_position_index[1:, :] = relative_coords.sum(-1)
        relative_position_index[0, :] = self.num_relative_distance - 1

        self.register_buffer('relative_position_index',
                             relative_position_index)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def extract_kv(self,x):
        B, N, C = x.shape
        if self.q_bias is not None:
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            kv_bias = torch.cat(( k_bias, self.v_bias))
        kv = F.linear(input=x, weight=self.kv.weight, bias=kv_bias)
        kv = kv.reshape(B, N, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        return k,v

    def forward(self, x, kv, ind=None):
        B, N, C = x.shape
        q = F.linear(input=x, weight=self.q_.weight, bias=self.q_bias)
        q = q.reshape(B, N, self.num_heads, -1).permute( 0, 2, 1, 3)
        k,v =kv
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.relative_position_bias_table is not None:
            Laq,Lat=self.relative_position_index.shape
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.reshape(-1)].reshape(Laq,Lat, -1)
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            if attn.shape[-2]==1 and ind is not None:
                relative_position_bias=relative_position_bias[:,ind:ind+1]
            attn = attn + relative_position_bias.unsqueeze(0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class TransDecoder(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 attn_drop_rate=0.,
                 cross_attn_drop_rate=0.,
                 drop_path_rate=0.,
                 proj_drop_rate=0.,
                 cross_proj_drop_rate=0.,
                 num_fcs=2,
                 qv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 window_size=None,
                 init_values=None):
        super().__init__()
        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.self_attn = SelfAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            qv_bias=qv_bias,
            qk_scale=None,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)
        self.cross_attn=CrossAttention(
            embed_dims=embed_dims,
            embed_dims2=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            qv_bias=qv_bias,
            qk_scale=None,
            attn_drop_rate=cross_attn_drop_rate,
            proj_drop_rate=cross_proj_drop_rate,
            init_cfg=None
        )
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=0.,
            dropout_layer=None,
            act_cfg=act_cfg,
            add_identity=False)
        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, embed_dims, postfix=3)
        self.add_module(self.norm3_name, norm3)

        # NOTE: drop path for stochastic depth, we shall see if
        # this is better than dropout here
        dropout_layer = dict(type='DropPath', drop_prob=drop_path_rate)
        self.drop_path = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.gamma_1 = nn.Parameter(
            init_values * torch.ones((embed_dims)), requires_grad=True)
        self.gamma_2 = nn.Parameter(
            init_values * torch.ones((embed_dims)), requires_grad=True)
        self.gamma_3 = nn.Parameter(
            init_values * torch.ones((embed_dims)), requires_grad=True)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x,y=None,context=None,kv=None):
        ind=None
        if context is not None:
            ind=len(context[0])
        if kv is None:
            kv=self.cross_attn.extract_kv(y)
        x0,context=self.self_attn(self.norm1(x),context)
        x = x + self.drop_path(self.gamma_1 * x0)
        x = x + self.drop_path(self.gamma_2 * self.cross_attn(self.norm2(x),kv,ind))
        x = x + self.drop_path(self.gamma_2 * self.ffn(self.norm3(x)))
        return x,context,kv


@HEADS.register_module()
class ARTrans(BaseDecodeHead):

    def __init__(self,
                 in_index=[-1]*8,
                 channels=32,
                 dropout_ratio=0.1,
                 img_size=512,
                 patch_size=16,
                 decode_patch_size=4,
                 num_classes=150,
                 embed_dims=768,
                 num_layers=8,
                 num_heads=12,
                 mlp_ratio=4,
                 qv_bias=True,
                 attn_drop_rate=0.,
                 cross_attn_drop_rate=0.,
                 drop_path_rate=0.,
                 proj_drop_rate=0.,
                 cross_proj_drop_rate=0.,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 final_norm=False,
                 num_fcs=2,
                 init_values=0.1,
                 init_cfg=None):
        super().__init__(input_transform='multiple_select',channels=channels,dropout_ratio=dropout_ratio,in_index=in_index,in_channels=[embed_dims]*len(in_index),num_classes=num_classes)
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.decode_patch_size=decode_patch_size
        self.patch_embed = nn.Linear((num_classes+1)*patch_size*patch_size,embed_dims)
        self.patch_decoder=nn.Linear(embed_dims,channels*decode_patch_size*decode_patch_size)
        self.sg_bn=nn.BatchNorm2d(channels)
        window_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.patch_shape = window_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransDecoder(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    cross_attn_drop_rate=cross_attn_drop_rate,
                    drop_path_rate=dpr[i],
                    proj_drop_rate=proj_drop_rate,
                    cross_proj_drop_rate=cross_proj_drop_rate,
                    num_fcs=num_fcs,
                    qv_bias=qv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    window_size=window_size,
                    init_values=init_values))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        trunc_normal_(self.cls_token, std=.02)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    if 'ffn' in n:
                        nn.init.normal_(m.bias, mean=0., std=1e-6)
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', bias=0.)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                constant_init(m, val=1.0, bias=0.)

    def flatten(self,input):
        return einops.rearrange(input,"B C H W -> B (H W) C")


    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        kv = [l.cross_attn.extract_kv(self.flatten(i)) for i,l in zip(inputs,self.layers)]
        contexts=[[[],[]]]*len(self.layers)
        x=self.cls_token
        g=[]
        for j in range(self.patch_shape[0]*self.patch_shape[1]):
            for i in range(len(self.layers)):
                x,context,_=self.layers[i](x,context=contexts[i],kv=kv[i])
                contexts[i]=context
            x=einops.rearrange(self.patch_decoder(x),"B 1 (C H W) -> B C H W",H=self.decode_patch_size,W=self.decode_patch_size,C=self.channels)
            x=resize(x,(self.patch_size,self.patch_size))
            x=F.relu(self.sg_bn(x))
            x=self.cls_seg(x)
            g.append(x)
            x=einops.rearrange(x,"B C H W -> B (H W) C")
            x=torch.argmax(x,dim=-1)
            x=F.one_hot(x,self.num_classes+1).to(dtype=torch.float32)
            x=self.patch_embed(einops.rearrange(x,"B (H W) C -> B 1 (C H W)",H=self.patch_size,W=self.patch_size))
        g=torch.stack(g,dim=1)
        g=einops.rearrange(g,"B HW c h w -> B (c h w) HW")
        g=F.fold(g,self.img_size,(self.patch_size,self.patch_size),stride=(self.patch_size,self.patch_size))
        return g

    def forward_train_core(self,gt,kv):
        x=torch.cat([einops.repeat(self.cls_token,"b l c -> (r b) l c",r=gt.shape[0]),gt],dim=1)
        for i,j in zip(kv,self.layers):
            x,_,_=j(x,kv=i)
        x=x[:,:-1]
        x=self.patch_decoder(x)
        B=x.shape[0]
        x=einops.rearrange(x,"B HW (c h w) -> (B HW) c h w",h=self.decode_patch_size,w=self.decode_patch_size)
        x=resize(x,(self.patch_size,self.patch_size))
        x=F.relu(self.sg_bn(x))
        x=self.cls_seg(x)
        x=einops.rearrange(x,"(B HW) c h w -> B (c h w) HW",B=B)
        seg_logits=F.fold(x,self.img_size,(self.patch_size,self.patch_size),stride=(self.patch_size,self.patch_size))
        return seg_logits

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg=dict(num_iters=1)):
        #gt_seg B,1,H,W
        total_loss=[]
        inputs=self._transform_inputs(inputs)
        kv = [l.cross_attn.extract_kv(self.flatten(i)) for i,l in zip(inputs,self.layers)]
        gt=gt_semantic_seg.squeeze(1).clamp(0,self.num_classes)
        for i in range(train_cfg['num_iters']):
            gt=einops.rearrange(F.one_hot(gt,self.num_classes+1).to(dtype=torch.float32),"B H W C -> B C H W")
            gt=F.unfold(gt,(self.patch_size,self.patch_size),stride=(self.patch_size,self.patch_size))
            gt=self.patch_embed(einops.rearrange(gt,"B C HW -> B HW C"))
            seg_logits=self.forward_train_core(gt,kv)
            gt=torch.argmax(seg_logits.detach(),dim=1)
            losses = self.losses(seg_logits, gt_semantic_seg)
            total_loss.append(losses)
        return reduce_dict(total_loss)