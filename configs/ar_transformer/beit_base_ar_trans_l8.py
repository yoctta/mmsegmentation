_base_ = [
    '../_base_/datasets/ade20k_512x512_dong.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
_beit_init=dict(type='Pretrained', checkpoint='pretrain/peco_800/iter_160000.pth')
_decode_head=dict(
    type="ARTrans",
    in_index=[-1,-1,-2,-2,-3,-3,-4,-4],
    channels=32,
    dropout_ratio=0.1,
    img_size=512,
    patch_size=16,
    num_classes=150,
    embed_dims=768,
    num_layers=8,
    num_heads=12,
    mlp_ratio=4,
    qv_bias=True,
    attn_drop_rate=0.2,
    cross_attn_drop_rate=0.0,
    drop_path_rate=0.,
    norm_cfg=dict(type='LN'),
    act_cfg=dict(type='GELU'),
    final_norm=False,
    num_fcs=2,
)
model = dict(
    type='EncoderDecoder',
    init_cfg=_beit_init,
    backbone=dict(
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
        init_values=0.1),
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
    train_cfg=dict(num_iters=2),
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
find_unused_parameters=True