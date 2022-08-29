_base_ = [
    '../_base_/datasets/ade20k_512x512_dong.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
from copy import deepcopy
norm_cfg = dict(type='SyncBN', requires_grad=True)
_beit_init=dict(type='Pretrained', checkpoint='pretrain/peco_800/iter_160000.pth')
_segmentor_backbone=dict(
        type='BEiT',
        init_cfg = None,
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(3, 5, 7, 11),
        qv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        norm_eval=False,
        init_values=0.1)

_segencoder=dict(
    type='MaskEncoderCNN',
    in_channels=150,
    out_channels=[48, 96, 192, 384, 384],
    out_layers=[1,2,3,4],
)
_mixer=dict(
    type='MixerPyramid2',
    image_feature_dim=768,
    mask_feature_dims=[ 96, 192, 384, 384],
    embed_dim=768,
    rescales=[4, 2, 1, 0.5]
)

_decode_head=dict(
    type='UPerHead',
    in_channels=[768, 768, 768, 768],
    in_index=[0, 1, 2, 3],
    pool_scales=(1, 2, 3, 6),
    channels=768,
    dropout_ratio=0.1,
    num_classes=150,
    norm_cfg=norm_cfg,
    align_corners=False,
    loss_decode=dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
)

_segmentor=dict(
    type='EncoderDecoder',
    init_cfg=None,
    backbone=_segmentor_backbone,
    neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5]),
    decode_head=_decode_head
)

_segmentor_backbone2=deepcopy(_segmentor_backbone)
_segmentor_backbone2['init_cfg']=_beit_init

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type="Refine",
        ref_segmentor=_segmentor,
        rel_segmentor=_segmentor_backbone2,
        ref_seg_encoder=_segencoder,
        mixer=_mixer),
    decode_head=_decode_head,
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=3e-5,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)


