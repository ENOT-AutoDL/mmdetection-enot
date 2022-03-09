import logging
import os
from functools import partial
from typing import Callable
from typing import Tuple
from typing import Union

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from enot.autogeneration import TransformationParameters
from enot.autogeneration.search_variants_from_model import generate_pruned_search_variants_model
from enot.distributed import is_master
from enot.distributed import synchronize_model_with_checkpoint
from enot.latency import initialize_latency
from enot.latency import max_latency
from enot.latency import mean_latency
from enot.latency import median_latency
from enot.latency import min_latency
from enot.models import SearchSpaceModel
from enot.optimize import EnotFixedLatencySearchOptimizer
from enot.optimize import EnotPretrainOptimizer
from enot.optimize import build_enot_optimizer
from enot.utils.train import WarmupScheduler

from mmcv import Config
from mmdet.enot_tools.nas_tools import MMDetDatasetEnumerateWrapper
from mmdet.enot_tools.nas_tools import custom_coco_evaluation
from mmdet.enot_tools.nas_tools import parse_losses
from mmdet.enot_tools.nas_tools import train_collate_function
from mmdet.enot_tools.nas_tools import valid_collate_function
from mmdet.models import build_detector

__all__ = [
    'PHASE_NAMES',
    'create_enot_optimizer',
    'get_metrics_and_loss',
    'init_model',
    'init_res_search_iteration',
]

PHASE_NAMES = (
    'train',
    'pretrain',
    'multiresolution_pretrain',
    'search',
    'search_with_resolution',
    'tune',
    'tune_on_resolution',
)

Scheduler = torch.optim.lr_scheduler._LRScheduler

# DEFAULT_DESCRIPTORS = tuple(TransformationParameters(width_mult=mult) for mult in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0))
DEFAULT_DESCRIPTORS = tuple(TransformationParameters(width_mult=mult) for mult in (1.0, 0.5, 0.33, 0.17, 0.08, 0.0))

def init_dataloaders(
        cfg: Config,
        train_dataset: torch.utils.data.Dataset,
        valid_dataset: torch.utils.data.Dataset,
):
    # For multigpu we use distributed data sampler.
    use_distributed_sampler = dist.is_initialized()

    valid_dataloader = torch.utils.data.DataLoader(
        dataset=MMDetDatasetEnumerateWrapper(valid_dataset),
        batch_size=1,
        collate_fn=valid_collate_function,
    )

    sampler_train = DistributedSampler(train_dataset, shuffle=True) if use_distributed_sampler else None
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=train_collate_function,
        num_workers=cfg.data['workers_per_gpu'],
        sampler=sampler_train,
    )

    return train_dataloader, valid_dataloader


def init_and_log_latency(
        search_space: SearchSpaceModel,
        train_dataloader: torch.utils.data.DataLoader,
        logger: logging.Logger,
):
    sample_batch = next(iter(train_dataloader))

    search_space.original_model.real_forward = search_space.original_model.forward
    search_space.original_model.forward = search_space.original_model.forward_dummy
    latency_container = initialize_latency('mmac.thop', search_space, (sample_batch['img'].cuda(),))
    logger.info(f'operations_latencies: {latency_container._operations_latencies}')
    logger.info(f'Constant latency = {latency_container.constant_latency}\n'
                f'Min latency: {min_latency(latency_container)}\n'
                f'Mean latency: {mean_latency(latency_container)}\n'
                f'Max latency: {max_latency(latency_container)}\n'
                f'Median latency: {median_latency(latency_container)}\n')

    search_space.original_model.forward = search_space.original_model.real_forward


def build_search_space_from_model(
        model: torch.nn.Module,
        search_variant_descriptors: Tuple[TransformationParameters] = DEFAULT_DESCRIPTORS,
):
    model.backbone = generate_pruned_search_variants_model(model.backbone, search_variant_descriptors)

    search_space = SearchSpaceModel(model).cuda()
    return search_space


def create_scheduler(
        optimizer: Optimizer,
        epochs: int,
        len_train_loader: int,
        logger: logging.Logger,
        warmup_epochs: int,
):
    scheduler = CosineAnnealingLR(optimizer, T_max=len_train_loader * epochs, eta_min=1e-8)
    if warmup_epochs is not None:
        scheduler = WarmupScheduler(scheduler, warmup_steps=len_train_loader * warmup_epochs)
    else:
        logger.info(f'The number of warmup epochs is not specified.')
    return scheduler


def create_enot_optimizer(
        model: Union[SearchSpaceModel, torch.nn.Module],
        optimizer: Optimizer,
        phase_name: str,
        target_latency: Union[int, float] = None,
):
    if 'pretrain' in phase_name:
        enot_optimizer = EnotPretrainOptimizer(
            search_space=model,
            optimizer=optimizer,
        )
    elif 'search' in phase_name:
        enot_optimizer = EnotFixedLatencySearchOptimizer(
            search_space=model,
            optimizer=optimizer,
            max_latency_value=float(target_latency),
        )
    elif phase_name == 'tune_on_resolution':
        enot_optimizer = build_enot_optimizer(
            phase_name='tune',
            model=model,
            optimizer=optimizer,
        )
    else:
        enot_optimizer = build_enot_optimizer(
            phase_name=phase_name,
            model=model,
            optimizer=optimizer,
        )
    return enot_optimizer


def init_model(
        cfg: Config,
        logger: logging.Logger,
):
    """
    Initialize model for specified phase: search_space for pretrain and search, searched model for tune.
    Args:
        cfg: MMDetection config
        logger:

    Returns:

    """
    model = build_detector(cfg.model)

    if 'pretrain' in cfg.phase_name and hasattr(cfg, 'baseline_ckpt') and cfg.baseline_ckpt:
        logger.info(f'Use{cfg.baseline_ckpt} as baseline checkpoint')
        logger.info(
            model.load_state_dict(
                torch.load(cfg.baseline_ckpt)['state_dict']
            )
        )

    if cfg.phase_name in ('pretrain', 'multiresolution_pretrain',
                          'search', 'search_with_resolution',
                          'tune', 'tune_on_resolution'):
        model = build_search_space_from_model(model)

        if 'tune' in cfg.phase_name or 'search' in cfg.phase_name:
            logger.info(f"Start from SearchSpace checkpoint: {cfg.ss_checkpoint}")
            if hasattr(cfg, 'ss_checkpoint'):
                model.load_state_dict(
                    torch.load(
                        cfg.ss_checkpoint,
                        map_location='cpu',
                    )['model'],
                    strict=False,
                )
        if 'tune' in cfg.phase_name:
            if hasattr(cfg, 'searched_arch') and cfg.searched_arch:
                logger.info(f"Number of operations in cfg.searched_arch: {len(cfg.searched_arch)}")
                logger.info(f"Number of operations in model.search_blocks: {len(model.search_blocks)}")
                model = model.get_network_by_indexes(cfg.searched_arch)
            else:
                model = model.get_network_with_best_arch()

    if cfg.resume_from:
        logger.info(f'resume from {cfg.resume_from}')
        logger.info(
            model.load_state_dict(
                torch.load(
                    cfg.resume_from,
                    map_location="cpu",
                )["model"],
                strict=True,
            )
        )
    # User has to manually place model to cuda.
    model.cuda()
    # synchronize models weights for multigpu
    synchronize_model_with_checkpoint(model)

    return model


def init_res_search_iteration(
        r_step: int,
        resolution: int,
        logger: logging.Logger,
        exp_dir: str,
        optimizer: Optimizer,
        epochs: int,
        len_train_loader: int,
        warmup_epochs: int,
):
    logger.info(f'RESOLUTION_SEARCH_STEP #{r_step}')
    logger.info(f'CURRENT RESOLUTION: {resolution}')
    if is_master():
        exp_dir_loc = os.path.join(exp_dir, f'resolution_{resolution}')
        os.makedirs(exp_dir_loc, exist_ok=True)
        writer_loc = SummaryWriter(exp_dir_loc)
    else:
        writer_loc = None
    # We should re-create scheduler as it is not updated by the resolution searcher.
    scheduler = create_scheduler(
        optimizer=optimizer,
        epochs=epochs,
        len_train_loader=len_train_loader,
        logger=logger,
        warmup_epochs=warmup_epochs,
    )

    path_save = os.path.join(exp_dir_loc, f'epoch_last.pth')
    return scheduler, writer_loc, path_save


def get_metrics_and_loss(
        dataset,
        metric: str,
) -> Tuple[Callable, Callable]:
    metric_function = partial(
        custom_coco_evaluation,
        dataset=dataset,
        metric=metric,
    )
    loss_function = parse_losses
    return metric_function, loss_function
