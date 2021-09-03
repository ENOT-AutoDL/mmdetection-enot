from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
from mmdet.datasets import CustomDataset
from torch import nn
from torch.utils.data import Dataset

__all__ = ['train_collate_function', 'valid_collate_function',
           'custom_coco_evaluation', 'custom_train_forward_logic', 'custom_valid_forward_logic',
           'sort_and_normalize_results', 'parse_losses', 'MMDetDatasetEnumerateWrapper', ]


def train_collate_function(batch: List[dict], device: Optional[str] = None) -> Dict[str, Any]:
    """Collate function for train dataset.

    Parameters
    ----------
    batch:
        List of sample from TRAIN dataset object.
    device:
        CUDA device. Deprecated

    Returns
    -------
    :
        Tuple with Input dictionary for model and None (for compatibility with enot, where expected (img, label))
    """
    image_metas = []
    gt_bboxes = []
    gt_labels = []
    images = []
    if device is None:
        for sample in batch:
            image_metas.append(sample['img_metas'].data)
            gt_bboxes.append(sample['gt_bboxes'].data)
            gt_labels.append(sample['gt_labels'].data)
            images.append(sample['img'].data)
    else:
        device = torch.device(device)
        for sample in batch:
            image_metas.append(sample['img_metas'].data)
            gt_bboxes.append(sample['gt_bboxes'].data.to(device))
            gt_labels.append(sample['gt_labels'].data.to(device))
            images.append(sample['img'].data.to(device))

    images_tensor = torch.zeros((len(batch), *(images[0].shape)), dtype=torch.float)
    if device is not None:
        images_tensor = images_tensor.to(device)
    for i, img in enumerate(images):
        images_tensor[i] = img
    return {
        'img': images_tensor,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'img_metas': image_metas,
    }


def valid_collate_function(batch: List[Tuple[dict, int]], device: Optional[str] = None) -> Tuple[Dict[str, Any], List[int]]:
    """Collate function for validation dataset

    Parameters
    ----------
    batch:
        List of sample from VALIDATION dataset object.
    device:
        CUDA device. Deprecated

    Returns
    -------
    :
        Tuple with Input dictionary for model and None (for compatibility with enot, where expected (img, label))

    """
    image_metas = []
    images = []
    indices = []
    if device is None:
        for sample, index in batch:
            image_metas.append(sample['img_metas'][0].data)
            images.append(sample['img'][0].data)
            indices.append(index)
    else:
        device = torch.device(device)
        for sample, index in batch:
            image_metas.append(sample['img_metas'][0].data)
            images.append(sample['img'][0].data.to(device))
            indices.append(index)

    images_tensor = torch.zeros((len(batch), *(images[0].shape)), dtype=torch.float)
    if device is not None:
        images_tensor = images_tensor.to(device)
    for i, img in enumerate(images):
        images_tensor[i] = img
    return {'img': [images_tensor], 'img_metas': [image_metas], }, indices


def custom_valid_forward_logic(model: nn.Module, data: Dict[str, Any]) -> Any:
    """Custom forward logic for validation"""
    return model(return_loss=False, rescale=True, img=data['img'], img_metas=data['img_metas'])


def custom_train_forward_logic(model: nn.Module, data: Dict[str, Any]) -> Any:
    """Custom forward logic for training"""
    return model(
        data['img'],
        data['img_metas'],
        gt_bboxes=data['gt_bboxes'],
        gt_labels=data['gt_labels'],
    )


def sort_and_normalize_results(results: List[Tuple[torch.Tensor, List[int]]], delete_duplicates=True):
    """Sort results from enot. Results are pair (bboxes, img_id).
    Function return results compatible with evaluate method."""
    sorted_results = sorted(results, key=lambda x: x[1][0])  # sort by image_id

    to_delete = set()
    if delete_duplicates:
        for index in range(len(sorted_results) - 1):
            _, curr_original = sorted_results[index]
            _, next_original = sorted_results[index + 1]
            if curr_original[0] == next_original[0]:
                to_delete.add(index)

    new_results = []

    for index, (predicted, _) in enumerate(sorted_results):
        if index not in to_delete:
            new_results.append(predicted[0])

    return new_results


def custom_coco_evaluation(results: List[Any], dataset: CustomDataset) -> Optional[Dict[str, float]]:
    """Evaluate results using dataset.evaluate for different sampled architectures.

    Parameters
    ----------
    results:
        List of results for different architectures.
    dataset:
        mmdetection dataset with evaluate method.

    Returns
    -------
    :
        Average COCO metrics for sample architectures.
    """
    results = sort_and_normalize_results(results)

    arch_metrics = dataset.evaluate(results)

    return arch_metrics


def parse_losses(losses: dict) -> torch.Tensor:
    """parse Detector loss from forward. By default forward method for detector return dictionary of loss"""
    if not isinstance(losses, dict):
        return 0.0

    log_vars = dict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')

    loss = sum(_value for _key, _value in log_vars.items()
               if 'loss' in _key)
    return loss


class MMDetDatasetEnumerateWrapper(Dataset):
    """Dataset wrapper for adding samples indices"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        data = self.dataset[item]
        return data, item

    def __len__(self):
        return len(self.dataset)
