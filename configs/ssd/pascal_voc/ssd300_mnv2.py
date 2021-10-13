# model settings
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(2, 4, 6),
        widen_factor=1.0,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')
    ),
    neck=dict(
        type='SSDNeck',
        in_channels=(320,),
        out_channels=(320, 512, 256),
        level_strides=[2, 2],
        level_paddings=[1, 1],
        l2_norm_scale=None,
        use_depthwise=False,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU6')
    ),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(32, 96, 320, 512, 256),
        num_classes=20,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.001),

        # set anchor size manually instead of using the predefined
        # SSD300 setting.
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2, 0.6666, 1.5],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),

# model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True
