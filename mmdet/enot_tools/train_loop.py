import argparse
import logging
import os
import shutil
from argparse import Namespace
from functools import partial
from pathlib import Path
from typing import Callable
from typing import Union

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from enot.distributed import init_torch
from enot.distributed import is_master
from enot.distributed import torch_save
from enot.experimental.resolution_search import ConstantResolutionStrategy
from enot.experimental.resolution_search import PretrainResolutionStrategy
from enot.logging import prepare_log
from enot.models import SearchSpaceModel
from enot.optimize.base import BaseEnotOptimizer
from enot.utils.train import init_exp_dir

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.enot_tools.detection_resize_strategy import VOCResizeStrategy
from mmdet.enot_tools.resolution_searcher import ResolutionSearcher
from mmdet.enot_tools.utils_initialization import PHASE_NAMES, create_scheduler
from mmdet.enot_tools.utils_initialization import create_enot_optimizer
from mmdet.enot_tools.utils_initialization import get_metrics_and_loss
from mmdet.enot_tools.utils_initialization import init_and_log_latency
from mmdet.enot_tools.utils_initialization import init_dataloaders
from mmdet.enot_tools.utils_initialization import init_model
from mmdet.enot_tools.utils_initialization import init_res_search_iteration
from mmdet.enot_tools.utils_train_loop import train_loop

Scheduler = torch.optim.lr_scheduler._LRScheduler


def phase(
        model: Union[SearchSpaceModel, torch.nn.Module],
        epochs: int,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        optimizer: Optimizer,
        enot_optimizer: BaseEnotOptimizer,
        scheduler: Scheduler,
        metric_function: Callable,
        loss_function: Callable,
        exp_dir: str,
        logger: logging.Logger,
        eval_frequency: int = 1,
):
    if is_master():
        writer = SummaryWriter(exp_dir)
    else:
        writer = None

    if cfg.phase_name == 'multiresolution_pretrain':
        resize_detect = VOCResizeStrategy()
        resolution_strategy = PretrainResolutionStrategy(
            min_resolution=cfg.resolution_range[0],
            max_resolution=cfg.resolution_range[1],
            resize_function=partial(resize_detect, val=False),
        )
    elif cfg.phase_name == 'search_with_resolution':
        resize_detect = VOCResizeStrategy()

        resolution_strategy = partial(
            ConstantResolutionStrategy,
            resize_function=partial(resize_detect, val=False),
        )

        def sample_to_model_inputs(x):
            img = x['img']
            return (img,), {}

        # Create SearchResolutionWithFixedLatencyIterator object
        # for fixed resolution range and fixed target latency
        search_resolution_iter = ResolutionSearcher(
            enot_optimizer=enot_optimizer,
            dataloader=train_loader,
            latency_type='mmac.thop',
            min_resolution=cfg.resolution_range[0],
            max_resolution=cfg.resolution_range[1],
            resolution_tolerance=getattr(cfg, 'resolution_tolerance', 10),
            sample_to_model_inputs=sample_to_model_inputs,
            resolution_strategy_constructor=resolution_strategy,
        )

    if cfg.phase_name == 'search_with_resolution':
        best_resolution = 0
        for r_step, (resolution, resolution_strategy) in enumerate(search_resolution_iter):
            scheduler, writer_loc, path_save_sr = init_res_search_iteration(
                r_step=r_step,
                resolution=resolution,
                logger=logger,
                exp_dir=exp_dir,
                optimizer=optimizer,
                epochs=epochs,
                len_train_loader=len(train_loader),
                warmup_epochs=cfg.warmup_epochs,
            )
            train_loop(
                cfg=cfg,
                epochs=epochs,
                logger=logger,
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                enot_optimizer=enot_optimizer,
                metric_function=metric_function,
                loss_function=loss_function,
                scheduler=scheduler,
                writer=writer_loc,
                exp_dir=exp_dir,
                search_resolution_iter=search_resolution_iter,
                path_save_sr=path_save_sr,
                eval_frequency=eval_frequency,
            )
            best_resolution = search_resolution_iter.best_resolution

        logger.info(f'Best resolution is {best_resolution}')
    else:
        train_loop(
            cfg=cfg,
            epochs=epochs,
            logger=logger,
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            enot_optimizer=enot_optimizer,
            metric_function=metric_function,
            loss_function=loss_function,
            scheduler=scheduler,
            writer=writer,
            exp_dir=exp_dir,
            resolution_strategy=resolution_strategy if cfg.phase_name == 'multiresolution_pretrain' else None,
            eval_frequency=eval_frequency,
        )


def run_phase(
        cfg: Config,
        experiment_args: Namespace,
) -> None:
    """
    Using mmdetection config to build model and dataset and start phase.
    """
    if cfg.phase_name not in PHASE_NAMES:
        raise AttributeError(f'Unknown phase_name: {cfg.phase_name}')

    exp_dir = init_exp_dir(experiment_args.rank, Path(cfg.work_dir))
    init_torch(cfg.random_seed, experiment_args.local_rank)
    logger = prepare_log(log_path=exp_dir / f'log_{cfg.phase_name}.txt')

    # Now supported only single dataset configs.
    train_dataset = build_dataset(cfg.data.train)
    # is important for valid metrics
    cfg.data.test.test_mode = True
    valid_dataset = build_dataset(cfg.data.test)

    train_dataloader, valid_dataloader = init_dataloaders(
        cfg=cfg,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    )

    logger.info('Dataloaders are ready')

    # Initialize model
    model = init_model(
        cfg=cfg,
        logger=logger,
    )
    logger.info("Model is ready")

    if 'search' in cfg.phase_name:
        init_and_log_latency(search_space=model, train_dataloader=train_dataloader, logger=logger)

    # For pretrain phase we update SearchSpace params.
    if cfg.phase_name in ('train', 'tune', 'tune_on_resolution'):
        params = model.parameters()
    elif 'pretrain' in cfg.phase_name:
        params = model.model_parameters()
    elif 'search' in cfg.phase_name:
        params = model.architecture_parameters()

    optimizer_constructor = getattr(torch.optim, cfg.optimizer['type'])
    optimizer = optimizer_constructor(
        params=params,
        **cfg.optimizer_params
    )
    scheduler = create_scheduler(
        optimizer=optimizer,
        epochs=cfg.total_epochs,
        len_train_loader=len(train_dataloader),
        logger=logger,
        warmup_epochs=cfg.warmup_epochs,
    )

    enot_optimizer = create_enot_optimizer(
        model=model,
        optimizer=optimizer,
        phase_name=cfg.phase_name,
        target_latency=cfg.target_latency if 'search' in cfg.phase_name else None,
    )

    logger.info('Train schedule is ready')

    torch_save(
        {
            'epoch': 0,
            'model': model.state_dict(),
        },
        os.path.join(exp_dir, 'checkpoint-0.pth'),
    )

    metric_function, loss_function = get_metrics_and_loss(
        dataset=valid_dataset,
        metric=cfg.evaluation.metric,
    )

    phase(
        model=model,
        epochs=cfg.total_epochs,
        train_loader=train_dataloader,
        validation_loader=valid_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        enot_optimizer=enot_optimizer,
        metric_function=metric_function,
        loss_function=loss_function,
        exp_dir=exp_dir,
        logger=logger,
        eval_frequency=cfg.eval_freq,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rank',
        help='Process rank',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--local_rank',
        help='Process rank',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--config-path',
        help='Path to train config',
        type=str,
        default=None,
    )

    command_line_args = parser.parse_args()

    cfg = Config.fromfile(command_line_args.config_path)
    # Copy runner
    os.makedirs(cfg.work_dir, exist_ok=True)
    file_name = Path(__file__).name
    shutil.copy(__file__, f'{cfg.work_dir}/{os.getpid()}_{file_name}')
    # Copy mmdet cfg file.
    cfg.dump(f'{cfg.work_dir}/{os.getpid()}_mmdetcfg.py')

    if cfg.optimizer['type'] in ('Adam', 'AdamW'):
        cfg.optimizer_params.pop('momentum')

    run_phase(
        cfg=cfg,
        experiment_args=command_line_args,
    )
