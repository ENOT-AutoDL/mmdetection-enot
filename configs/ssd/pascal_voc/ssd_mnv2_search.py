_base_ = [
    '../../ssd/pascal_voc/ssd_mnv2_train.py',
]

total_epochs = 20

batch_size = 20
use_batchnorm_tuning = True
data = dict(samples_per_gpu=batch_size)

# optimizer
optimizer = dict(type='SGD', lr=1., momentum=0.9, weight_decay=0)
optimizer_params = dict(lr=1., momentum=0.9, weight_decay=0)
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
resume_from = './workdir/multires_pretrain_our_2/epoch_59.pth'
work_dir = 'workdir/search_test'

random_seed = 0

scheduler = dict(type='CosineAnnealingLR')
scheduler_params = dict(eta_min=8e-6)
warmup_epochs = 1
eval_freq = 1

phase_name = 'search'
target_latency = 830