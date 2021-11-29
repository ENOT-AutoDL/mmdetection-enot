import copy
import logging
import os
from typing import Callable
from typing import Union

import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from enot.distributed import is_master
from enot.distributed import torch_save
from enot.experimental.resolution_search.resolution_strategy import ResolutionStrategy
from enot.models import SearchSpaceModel
from enot.optimize.base import BaseEnotOptimizer
from enot.utils.train import create_checkpoint

from mmcv import Config
from mmdet.core import encode_mask_results
from mmdet.enot_tools.nas_tools import custom_train_forward_logic
from mmdet.enot_tools.nas_tools import custom_valid_forward_logic
from mmdet.enot_tools.resolution_searcher import ResolutionSearcher

__all__ = [
    'train_loop',
]

Scheduler = torch.optim.lr_scheduler._LRScheduler


def evaluate(
        cfg: Config,
        model: Union[SearchSpaceModel, torch.nn.Module],
        validation_loader: DataLoader,
        bn_tune_loader: DataLoader,
        enot_optimizer: BaseEnotOptimizer,
        metric_function: Callable,
        loss_function: Callable,
        epoch: int,
        logger: logging.Logger,
        writer: SummaryWriter,
        resolution_strategy: ResolutionStrategy = None,
):
    model.eval()
    if 'search' in cfg.phase_name:
        enot_optimizer.prepare_validation_model()
    validation_loss = 0

    collected_outputs = list()
    use_test_model = 'pretrain' in cfg.phase_name or 'search' in cfg.phase_name
    if use_test_model:
        if 'pretrain' in cfg.phase_name:
            arch_to_test = [0] * len(model.search_variants_containers)
            test_model = model.get_network_by_indexes(arch_to_test)
            test_model.eval()
        else:
            test_model = model.get_network_with_best_arch()
            test_model.eval()

    if getattr(cfg, 'use_batchnorm_tuning', None):
        state = copy.deepcopy(model.state_dict())
        logger.info(f'BatchNorm tuning')
        test_model.train() if use_test_model else model.train()
        for data in tqdm(bn_tune_loader):
            with torch.no_grad():
                _ = custom_train_forward_logic(test_model if use_test_model else model, data)
        test_model.eval() if use_test_model else model.eval()

    val_loader = validation_loader if resolution_strategy is None else resolution_strategy(validation_loader)
    for data, indices in tqdm(val_loader):
        with torch.no_grad():
            losses = custom_valid_forward_logic(
                test_model if use_test_model else model,
                data,
            )
            if isinstance(losses[0], tuple):  # in case of instance segmentation
                losses = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in losses]
            batch_loss = loss_function(losses)
            validation_loss += batch_loss
            collected_outputs.append((losses, indices))

    validation_loss /= len(validation_loader)
    current_metrics = metric_function(collected_outputs)

    if is_master():
        logger.info(f'epoch: {epoch}')
        logger.info(f'validation metrics: {current_metrics}')
        logger.info(f'validation_loss: {validation_loss}')
        if 'search' in cfg.phase_name:
            logger.info(f'latency: {model.forward_latency.item()}')
            logger.info(f'arch_best {model.best_architecture_int}')
            logger.info(f'arch_probs: {model.architecture_probabilities}')

        for metric_name, metric_value in current_metrics.items():
            writer.add_scalar(f'val/{metric_name}', metric_value, epoch)
        writer.flush()
    if getattr(cfg, 'use_batchnorm_tuning', None):
        model.load_state_dict(state_dict=state)
    return current_metrics.get('mAP', 0.0)


def train_epoch(
        cfg: Config,
        model: Union[SearchSpaceModel, torch.nn.Module],
        epoch: int,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        enot_optimizer: BaseEnotOptimizer,
        metric_function: Callable,
        loss_function:  Callable,
        scheduler: Scheduler,
        logger: logging.Logger,
        writer: SummaryWriter,
        resolution_strategy: ResolutionStrategy = None,
):
    model.train()
    train_loss = 0
    n = 0
    train_loader_rs = train_loader if resolution_strategy is None else resolution_strategy(train_loader)
    # test default (zero) architecture in search_space
    if epoch == 0:
        evaluate(
            cfg=cfg,
            model=model,
            validation_loader=validation_loader,
            enot_optimizer=enot_optimizer,
            metric_function=metric_function,
            loss_function=loss_function,
            epoch=epoch,
            logger=logger,
            writer=writer,
            bn_tune_loader=train_loader,
        )
    logger.info(f'len(train_loader) {len(train_loader)}')
    for idx, data in enumerate(tqdm(train_loader_rs)):
        model.train()
        if isinstance(model, SearchSpaceModel) and not model.output_distribution_optimization_enabled:
            # Otherwise, an "unexpected argument" error occurs.
            if 'gt_masks' in data:
                model.initialize_output_distribution_optimization(
                    data['img'],
                    data['img_metas'],
                    gt_bboxes=data['gt_bboxes'],
                    gt_labels=data['gt_labels'],
                    gt_masks=data['gt_masks'],
                )
            else:
                model.initialize_output_distribution_optimization(
                    data['img'],
                    data['img_metas'],
                    gt_bboxes=data['gt_bboxes'],
                    gt_labels=data['gt_labels'],
                )

        def closure():
            nonlocal n
            nonlocal train_loss

            enot_optimizer.zero_grad()
            losses = custom_train_forward_logic(model, data)
            batch_loss = loss_function(losses)
            if 'search' in cfg.phase_name:
                batch_loss = enot_optimizer.modify_loss(batch_loss)
            batch_loss.backward()
            train_loss += batch_loss.item()
            n += 1

        enot_optimizer.step(closure)

        if scheduler is not None:
            scheduler.step()

        if is_master():
            writer.add_scalar(
                'train/step_lr',
                enot_optimizer._optimizer.param_groups[0]['lr'],
                epoch * len(train_loader) + idx,
            )
    train_loss /= n

    if is_master():
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.flush()
        logger.info(f'train_loss: {train_loss}')


def train_loop(
        cfg: Config,
        epochs: int,
        logger: logging.Logger,
        model: Union[SearchSpaceModel, torch.nn.Module],
        train_loader: DataLoader,
        validation_loader: DataLoader,
        enot_optimizer: BaseEnotOptimizer,
        metric_function: Callable,
        loss_function: Callable,
        scheduler: Scheduler,
        writer: SummaryWriter,
        exp_dir: str,
        resolution_strategy: ResolutionStrategy = None,
        search_resolution_iter: ResolutionSearcher = None,
        path_save_sr: str = None,
        eval_frequency: int = 1,
):

    for epoch in range(epochs):
        logger.info(f'EPOCH #{epoch}')

        train_epoch(
            cfg=cfg,
            model=model,
            epoch=epoch,
            train_loader=train_loader,
            validation_loader=validation_loader,
            enot_optimizer=enot_optimizer,
            metric_function=metric_function,
            loss_function=loss_function,
            scheduler=scheduler,
            logger=logger,
            writer=writer,
            resolution_strategy=resolution_strategy,
        )

        checkpoint = create_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=enot_optimizer._optimizer,
            scheduler=scheduler,
        )
        if cfg.phase_name == 'search_with_resolution':
            path_save = path_save_sr
        else:
            path_save = os.path.join(exp_dir, f'epoch_{epoch}.pth')
        print(f'path_save = {path_save}')
        torch_save(
            checkpoint,
            path_save,
        )
        if epoch % eval_frequency != 0:
            continue
        validation_metrics = evaluate(
            cfg=cfg,
            model=model,
            validation_loader=validation_loader,
            bn_tune_loader=train_loader,
            enot_optimizer=enot_optimizer,
            metric_function=metric_function,
            loss_function=loss_function,
            epoch=epoch,
            logger=logger,
            writer=writer,
        )
    if cfg.phase_name == 'search_with_resolution':
        search_resolution_iter.set_resolution_target_metric(validation_metrics)
        best_resolution = search_resolution_iter.best_resolution
        logger.info(f'best_resolution = {best_resolution}')
        checkpoint['best_resolution'] = best_resolution
        torch_save(
            checkpoint,
            os.path.join(exp_dir, f'epoch_{epoch}.pth')
        )
