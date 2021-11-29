from copy import copy
from copy import deepcopy
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch.nn import functional as F

DataSampleType = Dict[str, torch.Tensor]
ValidationSampleType = Tuple[DataSampleType, List[int]]
TrainSampleType = Dict[str, torch.Tensor]


class VOCResizeStrategy:
    def __init__(
            self,
            keep_ratio: bool = False,
            mode: str = 'bicubic',
    ):
        """
        Resize strategy is Callable object that contains the logic about resizing batch of data.
        It is used as a parameter for PretrainResolutionStrategy
        Args:
            mode: interpolation mode, see torch.nn.functional.interpolate for more details.
        """
        self._keep_ratio = keep_ratio
        self._resize_function = partial(F.interpolate, mode=mode)

    def resize_train(
            self,
            data: TrainSampleType,
            size: int
    ):
        images_tensor = data['img']
        data['img'] = self._resize_function(images_tensor, size)

        w_scale = size / images_tensor.shape[-1]
        h_scale = size / images_tensor.shape[-2]
        scale_factors = torch.tensor([[w_scale, h_scale, w_scale, h_scale]],
                                     dtype=torch.float32).cuda()

        scaled_bboxes = []
        for bbox in data['gt_bboxes']:
            scaled_bbox = bbox * scale_factors
            scaled_bbox[:, 0::2] = torch.clamp(scaled_bbox[:, 0::2], 0, data['img'].shape[-1])
            scaled_bbox[:, 1::2] = torch.clamp(scaled_bbox[:, 1::2], 0, data['img'].shape[-2])

            scaled_bboxes.append(scaled_bbox)

        data['gt_bboxes'] = scaled_bboxes

        for meta in data['img_metas']:
            self._update_meta(meta_data=meta, size=size)

        return data

    def _update_meta(
            self,
            meta_data: Dict[str, Any],
            size: int,
    ):
        w_scale = size / meta_data['ori_shape'][0]
        h_scale = size / meta_data['ori_shape'][1]
        meta_data['img_shape'] = (size, size, 3)
        meta_data['pad_shape'] = (size, size, 3)
        meta_data['scale_factor'] = np.array([w_scale, h_scale, w_scale, h_scale],
                                             dtype=np.float32)

    def resize_val(
            self,
            data: ValidationSampleType,
            size: int
    ):
        data, index = data

        result = copy(data)
        images_tensor = data['img'][0]
        result['img'][0] = self._resize_function(images_tensor, size)

        for meta in result['img_metas'][0]:
            self._update_meta(meta_data=meta, self=size)

        result = (result, index)
        return result

    def __call__(
            self,
            data: Union[ValidationSampleType, TrainSampleType],
            size: int,
            val=False
    ):
        resize_function = self.resize_val if val else self.resize_train
        return resize_function(data, size)
