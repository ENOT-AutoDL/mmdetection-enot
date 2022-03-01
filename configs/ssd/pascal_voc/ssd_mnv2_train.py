_base_ = [
    'ssd_mnv2.py',
    '../../_base_/datasets/voc0712.py',
    '../../_base_/schedules/schedule_2x.py',
    '../../_base_/default_runtime.py',
]

total_epochs = 100
batch_size = 24

phase_name = 'train'
# dataset settings
resolution = 480

# need to scale data to max resolution because of details of implementation of resolution search
img_scale = (resolution, resolution)
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset', times=1, dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=5e-4)
optimizer_params = dict(lr=1e-2, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 20])
checkpoint_config = dict(interval=1)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

work_dir = 'workdir/train_our/'
random_seed = 0

scheduler = dict(type='CosineAnnealingLR')
scheduler_params = dict(eta_min=8e-6)
warmup_epochs = 1
eval_freq = 1
