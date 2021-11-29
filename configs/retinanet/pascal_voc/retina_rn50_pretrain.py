_base_ = [
    '../../retinanet/pascal_voc/retina_rn50.py',
]

total_epochs = 100

batch_size = 35
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

baseline_ckpt = None  #'/4TB_second/prikhodko/ssd/projects/enot/mmdet/detection_demo/baseline/epoch_24.pth'
work_dir = '/4TB_second/prikhodko/ssd/projects/enot/mmdet/detection_demo/res_112_300_pretrain_retina_enot25/'

# baseline_ckpt = '/srv/scherbin/mmdet/detection_demo/baseline/epoch_24.pth'
# work_dir = '/srv/scherbin/mmdet/detection_demo/res_112_300_pretrain_retina_enot25/'
random_seed = 0

scheduler = dict(type='CosineAnnealingLR')
scheduler_params = dict(eta_min=8e-6)
warmup_epochs = 0
eval_freq = 1

phase_name = 'pretrain'
