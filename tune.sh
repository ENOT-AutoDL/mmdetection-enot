python=python
dist=torch.distributed.launch
executable=mmdet/enot_tools/train_loop.py
config_path=configs/ssd/pascal_voc/ssd_mnv2_tune.py

CUDA_VISIBLE_DEVICES=3 $python -m $dist --nproc_per_node=1 $executable --config-path=$config_path