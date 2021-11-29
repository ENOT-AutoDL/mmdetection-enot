_base_ = [
    '../../retinanet/pascal_voc/retina_rn50.py',
]

total_epochs = 40

batch_size = 42
data = dict(samples_per_gpu=batch_size)

# optimizer
optimizer = dict(type='SGD', lr=5e-5, momentum=0.9, weight_decay=5e-4)
optimizer_params = dict(lr=5e-5, momentum=0.9, weight_decay=5e-4)
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

ss_checkpoint = '/srv/scherbin/mmdet/detection_demo/res_112_300_pretrain_retina_enot25/epoch_99.pth'
searched_arch = [3, 2, 3, 2, 3, 4, 3, 4, 4, 3, 3, 4, 3, 4, 3, 3, 0, 0]
work_dir = f'/srv/scherbin/mmdet/detection_demo/tune_300_with_bn_train'
random_seed = 0

scheduler = dict(type='CosineAnnealingLR')
scheduler_params = dict(eta_min=8e-6)
warmup_epochs = 1
eval_freq = 1

phase_name = 'tune'
