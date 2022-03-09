config_path=configs/ssd/pascal_voc/ssd_mnv2_train.py

dist=torch.distributed.launch

CUDA_VISIBLE_DEVICES=0 python -m $dist --nproc_per_node=1 tools/train.py $config_path