_base_ = [
    'yolov3_d53_multiresolution_pretrain.py',
]

total_epochs = 10
batch_size = 16

phase_name = 'search_with_resolution'

target_latency = 42000
resolution_tolerance = 8

baseline_ckpt = None
resume_from = './work_dir/pascal_voc/multires_pretrain/checkpoint-0.pth'
work_dir = './work_dir/pascal_voc/search_with_resolution'


data = dict(samples_per_gpu=batch_size,)
# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.0, weight_decay=0)
optimizer_params = dict(lr=1e-3, momentum=0.0, weight_decay=0)
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

random_seed = 0

scheduler = dict(type='CosineAnnealingLR')
scheduler_params = dict(eta_min=8e-6)
warmup_epochs = 1
eval_freq = 1
