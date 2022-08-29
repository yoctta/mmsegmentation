_base_ = [
     '../_base_/datasets/ade20k_512x512_dong.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    type='GaussianDiffusionEncoderDecoder',
    backbone=dict(
        type='BEiT',
        init_cfg=dict(type='Pretrained', checkpoint='pretrain/peco_800/iter_160000.pth'),
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
    mask_backbone=dict(
        type='MaskEncoderTrans',
        img_size=(512, 512),
        patch_size=16,
        in_channels=150,  
        embed_dims=768,
        num_layers=4,
        num_heads=12,
        mlp_ratio=4,
        out_indices=(-1,-1,-1,-1)
    ),
    mixer=dict(
        type='MixerPyramid',
        image_feature_dim=768,
        mask_feature_dim=768,
        embed_dim=768,
        rescales=[4, 2, 1, 0.5]
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=768,
        dropout_ratio=0.1,
        num_classes=300,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(426, 426)),
    diffusion_cfg=dict(
        sched_name="squaredcos_cap_v2",
        diffusion_step=1000,
        model_mean_type="EPSILON",
        model_var_type="LEARNED_RANGE",
        loss_type="MSE",
        ignore_class=255,
        num_classes=150,
        timestep_respacing="ddim20"
    )
    )


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
data = dict(samples_per_gpu=1)
