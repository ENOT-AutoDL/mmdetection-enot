_base_ = '../mask_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

baseline_ckpt = None
ss_checkpoint = './maskrcnn_coco_search/epoch_0.pth'
searched_arch = [3, 2, 3, 2, 3, 2, 3, 2, 2, 3, 3, 2, 3, 2, 3, 3, 0, 0, 1, 2, 0, 1, 2, 1, 2, 1, 2, 1, 2, 3, 0, 1, 2]
phase_name = 'tune'
work_dir = './maskrcnn_coco_tune/'
total_epochs = 30
random_seed = 0
eval_freq = 1
batch_size = 2

# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer_params = dict(lr=1e-3, momentum=0.9, weight_decay=5e-4)
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