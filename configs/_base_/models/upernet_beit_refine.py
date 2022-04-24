from copy import deepcopy
norm_cfg = dict(type='SyncBN', requires_grad=True)
_beit_init=dict(type='Pretrained', checkpoint='pretrain/upernet_beit-base_8x2_640x640_160k_ade20k-eead221d.pth')
_segmentor_backbone=dict(
        type='BEiT',
        init_cfg = None,
        img_size=(640, 640),
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

_neck=dict(type='Feature2Pyramid', embed_dim=768, rescales=[4, 2, 1, 0.5])

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
    init_cfg=_beit_init,
    backbone=_segmentor_backbone,
    neck=_neck,
    decode_head=_decode_head
)

_segmentor_backbone2=deepcopy(_segmentor_backbone)
#_segmentor_backbone2['init_cfg']=_beit_init

model = dict(
    type='EncoderDecoder',
    init_cfg=_beit_init,
    backbone=dict(
        type="Refine",
        ref_segmentor=_segmentor,
        rel_segmentor=_segmentor_backbone2,
        ref_seg_encoder=dict(dim=150,out_dim=256,p_drop=0.5),
        mixer=dict(dims=[[768,256,768],[768,256,768],[768,256,768],[768,256,768]])),
    neck=_neck,
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
    test_cfg=dict(mode='whole'))
