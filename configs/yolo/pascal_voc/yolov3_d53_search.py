_base_ = [
    'yolov3_d53_train.py',
]

total_epochs = 20

batch_size = 8
use_batchnorm_tuning = True
data = dict(samples_per_gpu=batch_size)

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.0, weight_decay=0)
optimizer_params = dict(lr=0.001, momentum=0.0, weight_decay=0)
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

baseline_ckpt = None
resume_from = './work_dir/pascal_voc/pretrain/checkpoint-0.pth'
work_dir = 'work_dir/pascal_voc/search_test'

random_seed = 0

scheduler = dict(type='CosineAnnealingLR')
scheduler_params = dict(eta_min=8e-6)
warmup_epochs = 1
eval_freq = 1

phase_name = 'search'
target_latency = 26000