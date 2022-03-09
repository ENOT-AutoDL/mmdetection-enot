_base_ = [
    'yolov3_d53_train.py',
]

total_epochs = 100
batch_size = 8

# resolution = 608
# img_scale = (resolution, resolution)

data = dict(
    samples_per_gpu=batch_size,)

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

baseline_ckpt = 'work_dir/train/yolov3_d53_608_coco.pth'
work_dir = 'work_dir/pretrain/'

random_seed = 0

scheduler = dict(type='CosineAnnealingLR')
scheduler_params = dict(eta_min=8e-6)
warmup_epochs = 0
eval_freq = 1

phase_name = 'pretrain'
