config_path=configs/ssd/pascal_voc/ssd_mnv2_search_with_resolution.py

python=python
executable=mmdet/enot_tools/train_loop.py

CUDA_VISIBLE_DEVICES=0 python $executable --config-path=$config_path
