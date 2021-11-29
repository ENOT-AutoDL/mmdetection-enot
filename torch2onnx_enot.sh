CUDA_VISIBLE_DEVICES=2 python mmdet/enot_tools/pytorch2onnx_enot.py \
    configs/ssd/pascal_voc/ssd300_tune_on_resolution.py \
    /2tb/prikhodko/projects/enot/mmdet/logs/check/tune_on_resolution/epoch_0.pth \
    --output-file /2tb/prikhodko/projects/enot/mmdet/logs/check/onnx/ssd_mnv2_tuned.onnx \
    --input-img /home/prikhodko/projects/enot/mmdetection-enot/data/VOCdevkit/VOC2007/JPEGImages/000001.jpg\
    --shape 247 \
