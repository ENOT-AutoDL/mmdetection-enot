_base_ = [
    '../../retinanet/pascal_voc/retina_rn50.py',
]

total_epochs = 30

batch_size = 42
data = dict(samples_per_gpu=batch_size)

# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.0, weight_decay=5e-4)
optimizer_params = dict(lr=1e-3, momentum=0.0, weight_decay=5e-4)
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

ss_checkpoint = 'workdir/pretrain_retina_rn50/epoch_100.pth'
work_dir = 'workdir/search_retina_rn50'

random_seed = 0

scheduler = dict(type='CosineAnnealingLR')
scheduler_params = dict(eta_min=8e-6)
warmup_epochs = 1
eval_freq = 1

phase_name = 'search'
latency_loss_weight = 1e-5
