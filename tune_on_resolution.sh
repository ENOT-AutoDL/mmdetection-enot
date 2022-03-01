config_path=configs/ssd/pascal_voc/ssd_mnv2_tune_on_resolution.py

CUDA_VISIBLE_DEVICES=1 python mmdet/enot_tools/train_loop.py --config-path=$config_path
