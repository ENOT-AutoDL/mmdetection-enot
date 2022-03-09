_base_ = [
    'yolov3_d53_train.py',
]

total_epochs = 60
batch_size = 8
use_batchnorm_tuning = False


# сколько тюнить батчнорм
# 
batchnorm_tuning_iter_num = 200
use_batchnorm_tuning = True
# use_batchnorm_tuning = False

# width_mults = (1.0, 0.75, 0.5, 0.25, 0.0)

phase_name = 'multiresolution_pretrain'
resolution_range = (320, 608)

baseline_ckpt = './work_dir/train/yolov3_d53_608_coco.pth'

resume_from = './work_dir/multires_pretrain/epoch_14.pth'
work_dir = './work_dir/multires_pretrain_2'
random_seed = 0

# optimizer
optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=5e-4)
optimizer_params = dict(lr=5e-4, momentum=0.9, weight_decay=5e-4)
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

scheduler = dict(type='CosineAnnealingLR')
scheduler_params = dict(eta_min=8e-6)
warmup_epochs = 1
eval_freq = 1

img_scale = (resolution_range[1], resolution_range[1])
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

dataset_type = 'CocoDataset'
data_root = 'data/coco/'

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset', 
        times=1, 
        dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline)
    ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
