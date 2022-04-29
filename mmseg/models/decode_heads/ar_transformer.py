# Copyright (c) OpenMMLab. All rights reserved.
from turtle import forward
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

class TransDecoder(nn.Module):

    def __init__(self,channels,layers,cat_feats,) -> None:
        super().__init__()
        self.SOC=nn.Parameter(torch.zeros([1,1,self.channels]))
        self.blocks=nn.ModuleList([TransDecoderBlock(channels,num_heads,) for i in range(layers)])
    
    def forward(inputs):
        pass

@HEADS.register_module()
class ARTrans(BaseDecodeHead):

    def __init__(self, patch_size, layers, resolution, **kwargs):
        super().__init__(
            input_transform='multiple_select', **kwargs)
        self.patch_size=patch_size
        self.resolution=resolution
        self.patch_res=(resolution[0]//patch_size[0],resolution[1]//patch_size[1])
        self.transformer_decoder=TransDecoder(self.channels,layers)
        self.mask_patch_embedding= nn.Linear(self.num_classes*self.patch_size[0]*self.patch_size[1],self.channels)
        self.patch_decoder=nn.Linear(self.channels,self.num_classes*self.patch_size[0]*self.patch_size[1])
        self.dropout=nn.Dropout(self.dropout_ratio)

    

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        inputs = self._transform_inputs(inputs)
        t=
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

