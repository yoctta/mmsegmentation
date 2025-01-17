# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_beit(ckpt):
    new_ckpt = OrderedDict()
    is_bkb=False
    if 'backbone.blocks.0.attn.relative_position_bias_table' in ckpt.keys():
        print("is backbone")
        is_bkb=True
        ckpt_p1={i[9:]:ckpt[i] for i in ckpt if i.startswith('backbone')}
        ckpt_p2={i:ckpt[i] for i in ckpt if not i.startswith('backbone')}
    else:
        ckpt_p1=ckpt
        ckpt_p2={}
    print(len(ckpt.keys()),len(ckpt_p1.keys()),len(ckpt_p2.keys()))
    for k, v in ckpt_p1.items():
        if k.startswith('patch_embed'):
            print("convert ",k)
            new_key = k.replace('patch_embed.proj', 'patch_embed.projection')
        elif k.startswith('blocks'):
            print("convert ",k)
            new_key = k.replace('blocks', 'layers')
            if 'norm' in new_key:
                new_key = new_key.replace('norm', 'ln')
            elif 'mlp.fc1' in new_key:
                new_key = new_key.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in new_key:
                new_key = new_key.replace('mlp.fc2', 'ffn.layers.1')
        elif k.startswith('fpn1'):
            new_key = k.replace('fpn1', 'neck.upsample_4x')
        elif k.startswith('fpn2'):
            new_key = k.replace('fpn2', 'neck.upsample_2x')
        else:
            new_key = k
        if not new_key.startswith('neck') and is_bkb:
            new_key="backbone."+new_key
        new_ckpt[new_key] = v
    if is_bkb:
        for i in ckpt_p2:
            new_ckpt[i]=ckpt_p2[i]
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in official pretrained beit models to'
        'MMSegmentation style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    weight = convert_beit(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)


if __name__ == '__main__':
    main()
