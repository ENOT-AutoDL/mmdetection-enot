python=python
dist=torch.distributed.launch
executable=mmdet/enot_tools/train_loop.py
config_path=configs/retinanet/pascal_voc/retina_rn50_pretrain.py

CUDA_VISIBLE_DEVICES=2 $python -m $dist --nproc_per_node=1 $executable --config-path=$config_path
