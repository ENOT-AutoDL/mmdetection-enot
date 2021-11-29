python=python
#dist=torch.distributed.launch
executable=mmdet/enot_tools/train_loop.py
#config_path=/damage/vehicle_damage_detection/config/mmdet_mask-rcnn_r101_weighted_coco_7_cls_enot.py
config_path=configs/mask_rcnn/enot/mask_rcnn_r101_fpn_1x_coco_pretrain.py

#PYTHONPATH=/damage/vehicle_damage_detection:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 $python -m $dist --nproc_per_node=1 $executable --config-path=$config_path
PYTHONPATH=/damage/vehicle_damage_detection:$PYTHONPATH CUDA_VISIBLE_DEVICES=1 $python $executable --config-path=$config_path

