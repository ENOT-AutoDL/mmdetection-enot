config_path=configs/ssd/pascal_voc/ssd_mnv2_search.py

CUDA_VISIBLE_DEVICES=0 python mmdet/enot_tools/train_loop.py --config-path=$config_path
