import os.path as osp
import warnings

import torch.onnx
from mmcv import Config

from mmdet.enot_tools.train_loop import build_search_space_from_model
from mmdet.models import build_detector
from tools.deployment.pytorch2onnx import parse_args
from tools.deployment.pytorch2onnx import parse_normalize_cfg
from tools.deployment.pytorch2onnx import pytorch2onnx

if __name__ == '__main__':
    args = parse_args()
    warnings.warn('Arguments like `--mean`, `--std`, `--dataset` would be \
        parsed directly from config file and are deprecated and \
        will be removed in future releases.')

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    try:
        from mmcv.onnx.symbolic import register_extra_symbolics
    except ModuleNotFoundError:
        raise NotImplementedError('please update mmcv to version>=v1.0.4')
    register_extra_symbolics(args.opset_version)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.shape is None:
        img_scale = cfg.test_pipeline[1]['img_scale']
        input_shape = (1, 3, img_scale[1], img_scale[0])
    elif len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        if cfg.best_resolution is not None:
            spatial_size = cfg.best_resolution
            if spatial_size is tuple:
                input_shape = (1, 3) + spatial_size
            else:
                input_shape = (1, 3, spatial_size, spatial_size)
        else:
            raise ValueError('invalid input shape')

    model = build_detector(cfg.model)
    search_space = build_search_space_from_model(model)
    searched_model = search_space.get_network_by_indexes(cfg.searched_arch)

    searched_model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location='cpu',
        )['model'],
        strict=False,
    )
    searched_model.cpu()
    if not args.input_img:
        args.input_img = osp.join(osp.dirname(__file__), '../../demo/demo.jpg')

    normalize_cfg = parse_normalize_cfg(cfg.test_pipeline)

    # convert model to onnx file
    pytorch2onnx(
        searched_model,
        args.input_img,
        input_shape,
        normalize_cfg,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        test_img=args.test_img,
        do_simplify=args.simplify,
        dynamic_export=args.dynamic_export,
        skip_postprocess=args.skip_postprocess)
