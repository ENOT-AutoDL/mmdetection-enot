import argparse

import torch

from mmdet.apis import init_detector
from mmcv import Config
from enot.latency import MacCalculatorPthflops
from mmdet.enot_tools.utils_initialization import build_search_space_from_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-path',
        help='Path to train config',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--input-res',
        help='Resolution for input image',
        type=int,
        default=None,
    )

    args = parser.parse_args()

    cfg = Config.fromfile(args.config_path)

    if args.input_res is None:
        args.input_res = cfg.resolution

    model = init_detector(cfg)

    if hasattr(cfg, 'searched_arch'):
        search_space = build_search_space_from_model(model)
        model = search_space.get_network_by_indexes(cfg.searched_arch)

    model.real_forward = model.forward
    model.forward = model.forward_dummy

    # Calculator support only batch_size=1
    model_input = torch.ones(1, 3, args.input_res, args.input_res).cuda()
    mac_operations_number = MacCalculatorPthflops().calculate(model, model_input)

    print(mac_operations_number)
