config_path=configs/retinanet/pascal_voc/retina_rn50_tune.py

CUDA_VISIBLE_DEVICES=0 python mmdet/enot_tools/train_loop.py --config-path=$config_path
