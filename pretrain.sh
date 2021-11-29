python=python
dist=torch.distributed.launch
executable=mmdet/enot_tools/train_loop.py
config_path=configs/ssd/pascal_voc/ssd300_pretrain.py

CUDA_VISIBLE_DEVICES=2,3 $python -m $dist --nproc_per_node=2 $executable --config-path=$config_path
