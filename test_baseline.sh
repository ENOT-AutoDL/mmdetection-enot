config_path=configs/yolo/coco/yolov3_d53_train.py
chkpt_path=work_dir/train/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth
CUDA_VISIBLE_DEVICES=0 python tools/test.py $config_path  $chkpt_path --eval bbox