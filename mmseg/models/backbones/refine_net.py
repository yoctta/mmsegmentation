# Copyright (c) OpenMMLab. All rights reserved.
from turtle import forward
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed

try:
    from scipy import interpolate
except ImportError:
    interpolate = None

class SegEncoder(nn.Module):
    def __init__(self,dim,out_dim,p_drop) -> None:
        super().__init__()
        self.conv1=nn.Conv2d(dim,out_dim,1)
        self.bn1=nn.BatchNorm2d(out_dim)
        self.conv2=nn.Conv2d(dim,out_dim,3,padding=1)
        self.bn2=nn.BatchNorm2d(out_dim)
        self.conv3=nn.Conv2d(dim,out_dim,5,padding=2)
        self.bn3=nn.BatchNorm2d(out_dim)
        self.conv4=nn.Conv2d(3*dim,dim,1)
        self.dropout=nn.Dropout2d(p_drop)
        self.act=nn.GELU()
    def forward(self,seg_logits):
        x1=self.act(self.bn1(self.conv1(seg_logits)))
        x2=self.act(self.bn2(self.conv2(seg_logits)))
        x3=self.act(self.bn3(self.conv3(seg_logits)))
        n=self.dropout(self.conv4(self.cat([x1,x2,x3],1)))
        return tuple([n.clone() for i in range(4)])

@BACKBONES.register_module()
class Refine(BaseModule):
    def __init__(self,
                ref_segmentor,
                rel_segmentor,
                ref_seg_encoder,
                mixer):
        super(Refine, self).__init__()
        ref_segmentor=BACKBONES.build_backbone(ref_segmentor).eval()
        for param in ref_segmentor.parameters():
            param.requires_grad = False
        self.ref_segmentor=ref_segmentor
        self.ref_seg_encoder=SegEncoder(**ref_seg_encoder)
        self.rel_segmentor=BACKBONES.build_backbone(rel_segmentor)
        self.mixer_conv=nn.ModuleList([nn.Conv2d(i[0]+i[1],i[2],1,bias=False) for i in mixer['dims']])
        self.mixer_LN1=nn.ModuleList([nn.LayerNorm(i[0]) for i in mixer['dims']])
        self.mixer_LN2=nn.ModuleList([nn.LayerNorm(i[1]) for i in mixer['dims']])


    def forward(self, inputs):
        with torch.no_grad():
            ref_logits=torch.softmax(self.ref_segmentor(inputs),dim=1)
        features1=self.rel_segmentor(inputs)
        features2=self.ref_seg_encoder(ref_logits)
        features=[]
        for f1,f2,ln1,ln2,conv in zip(features1,features2,self.mixer_LN1,self.mixer_LN2,self.mixer_conv):
            features.append(conv(torch.cat([ln1(f1),nn.functional.interpolate(ln2(f2),f1.shape[2:],mode="bilinear")],1)))
        return tuple(features)
