# optimizer
lr = 0.000025
optimizer = dict(
    type='AdamW',
    lr=lr,
    betas=(0.95, 0.99),
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2., norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
