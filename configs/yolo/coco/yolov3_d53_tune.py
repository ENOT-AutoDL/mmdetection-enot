_base_ = [
    'yolov3_d53_train.py',
]

total_epochs = 40

batch_size = 8
data = dict(samples_per_gpu=batch_size)

# optimizer
optimizer = dict(type='AdamW', lr=1e-3, weight_decay=5e-4)
optimizer_params = dict(lr=1e-3, weight_decay=5e-4)
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

ss_checkpoint = None
searched_arch = [3, 1, 4, 5, 4, 4, 3, 4, 3, 2, 2, 3, 4, 4, 4, 4, 3, 4, 5, 5, 4, 4, 4, 4]
work_dir = 'work_dir/tune'
random_seed = 0
resume_from = 'work_dir/searched_tuned_x2/epoch_31.pth'

scheduler = dict(type='CosineAnnealingLR')
scheduler_params = dict(eta_min=8e-6)
warmup_epochs = 1
eval_freq = 1

phase_name = 'tune'
