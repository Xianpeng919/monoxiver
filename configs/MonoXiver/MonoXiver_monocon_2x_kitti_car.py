_base_ = [
    '../_base_/datasets/kitti-mono3d-car-monoxiver.py',
    '../_base_/schedules/adamw_2x.py',
    '../../external/mmdetection3d/configs/_base_/default_runtime.py'
]

model = dict(
    type='MonoXiver',
    pretrained=True,
    refine_only=True,
    backbone=dict(
        type='DLA', depth=34, norm_cfg=dict(type='BN')),
    neck=dict(
        type='DLAUp',
        in_channels_list=[64, 128, 256, 512],
        scales_list=(1, 2, 4, 8),
        start_level=2,
        norm_cfg=dict(type='BN')),
    rpn_head=dict(
        type='MonoConProposalHead',
        in_channel=64,
        feat_channel=64,
        num_classes=1,
        num_alpha_bins=12,
        use_AN=True,
    ),
    proposal_generator=dict(
        type='MonoXiverProposalGenerator',
        num_class=1,
        x_max=1.5,
        x_stride=0.75,
        z_max=1.5,
        z_stride=0.75,
    ),
    roi_head=dict(
        type='MonoXiverRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=64,
            featmap_strides=[4]),
        num_augments=25,
        bbox_head=dict(
            type='MonoXiverBboxHead',
            roi_feat_size=14,
            num_classes=1,
            in_channels=64,
            feat_channel=256,
            num_sa_layers=1,
            num_modes=4,
            num_heads=8,
            num_ffn_fcs=2,
            feedforward_channels=1024,
            ffn_act_cfg=dict(type='GELU'),
            dropout=0.0,
            num_cls_fcs=2,
            num_reg_fcs=2,
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_xyz=dict(type='L1Loss', loss_weight=5.0),
            loss_dim=dict(type='L1Loss', loss_weight=1.0),
        ),
    ),
    train_cfg=dict(
        rpn=dict(topk=50, local_maximum_kernel=3, thresh=0.01),
        rcnn=dict(
            assigner=dict(
                type='HungarianAssignerMono3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                iou_3d_cost=dict(type='IoU3DCost', iou_calculator=dict(type='BboxOverlaps3D', coordinate='camera'),
                                    weight=8.0)
            ),
            sampler=dict(type='PseudoSampler'),
            pos_weight=1,
            ),
    ),
    test_cfg=dict(
        rpn=dict(topk=30, local_maximum_kernel=3, thresh=0.4),
        rcnn=dict(
            score_thr_post=0.03,
            score_thr_final=0.15,
            topk=3,
        )
    )
)


checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
load_from = './ckpts/monocon_pretrained_rpn_car_only.pth'
evaluation = dict(interval=1)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
find_unused_parameters = True
