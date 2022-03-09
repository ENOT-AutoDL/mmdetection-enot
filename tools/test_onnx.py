import torch
import onnxruntime

import numpy as np

import argparse

import torch
from mmcv import Config
import mmcv
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('torch_checkpoint', help='checkpoint file')
    parser.add_argument('onnx_checkpoint', help='checkpoint file')
    args = parser.parse_args()
    return args

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

args = parse_args()

cfg = Config.fromfile(args.config)

# build the dataloader
dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_detector(cfg.model)
checkpoint = load_checkpoint(model, args.torch_checkpoint, map_location='cpu')

# eval model
model.eval()

# load onnx model
onnx_session = onnxruntime.InferenceSession(args.onnx_checkpoint)

prog_bar = mmcv.ProgressBar(len(dataset))
# compare onnx and pytorch outs
for i, data in enumerate(data_loader):
    x_eval = data['img'][0]

    with torch.no_grad():
        head_features = model.forward_dummy(x_eval)
        torch_outs = model.bbox_head.get_bboxes(*head_features, img_metas={}, rescale=False)
        
    onnx_input = {'input' : x_eval.numpy()}
    ort_outs = onnx_session.run(None, onnx_input)

    # get boundig boxes from outs
    # only for batch_size = 1
    torch_bboxes = torch_outs[0][0]
    ort_bboxes = ort_outs[0][0]

    # number of bounding boxes
    # torch model don't return bboxes with small confidence
    bboxes_num = torch_bboxes.shape[0]

    np.testing.assert_allclose(to_numpy(torch_bboxes), ort_bboxes[:bboxes_num], rtol=1e-03, atol=1e-03)
    
    batch_size = len(torch_outs)
    for _ in range(batch_size):
        prog_bar.update()