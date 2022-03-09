config_path=configs/yolo/coco/yolov3_d53_train.py
torch_chkpt_path=work_dir/train/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth
onnx_chkpt_path=work_dir/onnx/yolov3_d53_tuned.onnx

CUDA_VISIBLE_DEVICES=0 python tools/test_onnx.py $config_path  $torch_chkpt_path $onnx_chkpt_path
