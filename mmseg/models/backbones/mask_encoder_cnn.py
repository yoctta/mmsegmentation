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
from .beit import BEiTAttention
from mmseg.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import PatchEmbed
from ..necks.mixerpyramid import AdaNorm
try:
    from scipy import interpolate
except ImportError:
    interpolate = None

class ConvBnGelu(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,diffusion_step=0) -> None:
        super().__init__()
        self.conv=nn.Conv2d(in_channel,out_channel,3,stride,padding=1)
        self.bn=AdaNorm(out_channel,diffusion_step,"adabatchnorm_abs")
        self.gelu=nn.GELU()
    def forward(self,x,t=None):
        return self.gelu(self.bn(self.conv(x)))


@BACKBONES.register_module()
class MaskEncoderCNN(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 out_layers,
                 diffusion_step=0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.layers=nn.ModuleList([])
        self.out_layers=out_layers
        for i in out_channels:
            self.layers.append(ConvBnGelu(in_channels,i,2,diffusion_step))
            in_channels=i
    
    def forward(self,x,t=None):
        outs=[]
        for i,j in enumerate(self.layers):
            x=j(x)
            if i in self.out_layers:
                for _ in range(self.out_layers.count(i)):
                    outs.append(x)
        return tuple(outs)

