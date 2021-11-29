_base_ = [
    './ssd_lite_mnv2_mmdet_train.py',
]

total_epochs = 60
batch_size = 16
use_batchnorm_tuning = True

phase_name = 'multiresolution_pretrain'
resolution_range = (320, 640)

random_seed = 0

# optimizer
optimizer = dict(type='SGD', lr=3e-4, momentum=0.9, weight_decay=5e-4)
optimizer_params = dict(lr=3e-4, momentum=0.9, weight_decay=5e-4)
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

baseline_ckpt = 'workdir/train/epoch_99.pth'
resume_from = './workdir/multires_pretrain_our_2/epoch_59.pth'
work_dir = './workdir/multires_pretrain_our_3'
random_seed = 0

scheduler = dict(type='CosineAnnealingLR')
scheduler_params = dict(eta_min=8e-6)
warmup_epochs = 1
eval_freq = 1

img_scale = (resolution_range[1], resolution_range[1])
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
