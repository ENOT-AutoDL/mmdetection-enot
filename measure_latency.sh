config_path=configs/ssd/pascal_voc/ssd_mnv2_train.py

CUDA_VISIBLE_DEVICES=0 python mmdet/enot_tools/get_model_latency.py --config-path=$config_path