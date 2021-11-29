python=python
executable=mmdet/enot_tools/train_loop.py
config_path=configs/ssd/pascal_voc/ssd_mnv2_search_with_resolution.py

CUDA_VISIBLE_DEVICES=1 python $executable --config-path=$config_path

