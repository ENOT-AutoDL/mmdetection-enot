import argparse
from argparse import Namespace
from functools import partial
import logging
import os
from typing import Any
from typing import Tuple
from typing import Dict
from typing import Optional
from typing import Union
from typing import Callable
import shutil
from pathlib import Path

from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler
import torch.distributed as dist

from enot.latency import initialize_latency
from enot.logging import prepare_log
from enot.utils.data.dataloaders import CudaDataLoader
from enot.distributed import init_torch
from enot.distributed import synchronize_model_with_checkpoint
from enot.distributed import torch_save
from enot.utils.train import init_exp_dir
from enot.optimize import build_enot_optimizer
from enot.models import SearchSpaceModel
from enot.utils.train import WarmupScheduler
from enot.utils.train import create_checkpoint
from enot.autogeneration.search_variants_from_model import generate_pruned_search_variants_model
from enot.models import SearchVariantsContainer


from mmcv import Config
from mmcv.cnn import ConvModule

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.enot_tools.nas_tools import MMDetDatasetEnumerateWrapper
from mmdet.enot_tools.nas_tools import custom_coco_evaluation
from mmdet.enot_tools.nas_tools import custom_train_forward_logic
from mmdet.enot_tools.nas_tools import custom_valid_forward_logic
from mmdet.enot_tools.nas_tools import train_collate_function
from mmdet.enot_tools.nas_tools import parse_losses
from mmdet.enot_tools.nas_tools import valid_collate_function

Scheduler = torch.optim.lr_scheduler._LRScheduler


def evaluate(
        model: Union[SearchSpaceModel, torch.nn.Module],
        validation_loader: CudaDataLoader,
        validation_forward_wrapper: Callable,
        loss_function: Callable,
        metric_function: Callable,
        epoch: int,
        logger: logging.Logger,
        writer: SummaryWriter,
        sample_zero_arch: bool = False,
):
    model.eval()
    validation_loss = 0

    collected_outputs = list()
    if cfg.phase_name == 'pretrain':
        model.sample_random_arch()
    elif cfg.phase_name == 'search':
        if sample_zero_arch:
            model.sample([[0] for _ in model.search_blocks])
        else:
            model.sample_best_arch()
    for data, indices in tqdm(validation_loader, total=len(validation_loader)):
        with torch.no_grad():
            losses = validation_forward_wrapper(model, data)
            batch_loss = loss_function(losses)
            validation_loss += batch_loss
            collected_outputs.append((losses, indices))

    validation_loss /= len(validation_loader)
    current_metrics = metric_function(collected_outputs)

    if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
        logger.info(f'epoch: {epoch}')
        logger.info(f'validation metrics: {current_metrics}')
        logger.info(f'validation_loss: {validation_loss}')
        if cfg.phase_name == 'search':
            logger.info(f'latency: {model.forward_latency.item()}')
            logger.info(f'arch: {model.forward_architecture_int}')
            logger.info(f'arch_probs: {model.architecture_probabilities}')
        writer.add_scalar('val/mAP', current_metrics['mAP'], epoch)
        writer.flush()


def train(
        model: Union[SearchSpaceModel, torch.nn.Module],
        epochs: int,
        train_loader: CudaDataLoader,
        validation_loader: CudaDataLoader,
        enot_optimizer,
        scheduler: Scheduler,
        metric_function: Callable,
        loss_function: Callable,
        validation_forward_wrapper: Callable,
        train_forward_wrapper: Callable,
        exp_dir: str,
        logger: logging.Logger,
        eval_frequency: int,
):
    if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
        writer = SummaryWriter(exp_dir)
    else:
        writer = None

    for epoch in range(epochs):
        logger.info(f'EPOCH #{epoch}')

        model.train()
        global train_loss
        global train_accuracy
        global n
        train_loss = 0
        n = 0

        for idx, data in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
        ):
            # test default (zero) architecture in search_space
            if idx == 0 and epoch == 0 and cfg.phase_name in ('search', 'tune'):
                evaluate(
                    model=model,
                    validation_loader=validation_loader,
                    validation_forward_wrapper=validation_forward_wrapper,
                    loss_function=loss_function,
                    metric_function=metric_function,
                    epoch=epoch,
                    logger=logger,
                    writer=writer,
                    sample_zero_arch=True,
                )
                model.train()

            if isinstance(model, SearchSpaceModel) and not model.output_distribution_optimization_enabled:
                model.initialize_output_distribution_optimization(
                    data['img'],
                    data['img_metas'],
                    gt_bboxes=data['gt_bboxes'],
                    gt_labels=data['gt_labels'],
                )

            def closure():
                global n
                global train_loss
                global train_accuracy

                enot_optimizer.zero_grad()

                losses = train_forward_wrapper(model, data)
                batch_loss = loss_function(losses)
                if cfg.phase_name == 'search' and cfg.latency_loss_weight > 0:
                    batch_loss += model.loss_latency_expectation * cfg.latency_loss_weight
                batch_loss.backward()

                train_loss += batch_loss.item()
                n += 1

            enot_optimizer.step(closure)

            if scheduler is not None:
                scheduler.step()

            if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
                writer.add_scalar('train/step_lr', enot_optimizer._optimizer.param_groups[0]['lr'],
                                  epoch * len(train_loader) + idx)

        torch_save(
            create_checkpoint(
                epoch,
                model,
                enot_optimizer._optimizer,
                scheduler,
            ),
            os.path.join(exp_dir, f'epoch_{epoch}.pth')
        )

        train_loss /= n

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.flush()
            logger.info(f'train_loss: {train_loss}')

        if epoch % eval_frequency != 0:
            continue

        evaluate(
            model=model,
            validation_loader=validation_loader,
            validation_forward_wrapper=validation_forward_wrapper,
            loss_function=loss_function,
            metric_function=metric_function,
            epoch=epoch,
            logger=logger,
            writer=writer,
        )


def build_search_space_from_model(
        model: torch.nn.Module,
        width_multipliers: Tuple[float] = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0)
):
    model.backbone = generate_pruned_search_variants_model(model.backbone, width_multipliers)

    skip_conv_cls = ConvModule(
        model.bbox_head.in_channels,
        model.bbox_head.feat_channels,
        1,
        stride=1,
        padding=0,
        conv_cfg=model.bbox_head.conv_cfg,
        norm_cfg=model.bbox_head.norm_cfg,
    )

    skip_conv_reg = ConvModule(
        model.bbox_head.in_channels,
        model.bbox_head.feat_channels,
        1,
        stride=1,
        padding=0,
        conv_cfg=model.bbox_head.conv_cfg,
        norm_cfg=model.bbox_head.norm_cfg,
    )

    # Modify head for RetinaNet (it is hardcoded for now)
    # we need equivalent number of search variants in each container
    model.bbox_head.cls_convs = SearchVariantsContainer(
        [model.bbox_head.cls_convs] + [skip_conv_cls] * (len(width_multipliers) - 1))
    model.bbox_head.reg_convs = SearchVariantsContainer(
        [model.bbox_head.reg_convs] + [skip_conv_reg] * (len(width_multipliers) - 1))

    search_space = SearchSpaceModel(model).cuda()
    return search_space


def init_model(
        cfg: Config,
        is_search_space: bool,
        train_loader: CudaDataLoader,
        logger: logging.Logger,
):
    model = build_detector(cfg.model)
    if cfg.phase_name == 'pretrain' and cfg.baseline_ckpt:
        logger.info(
            model.load_state_dict(
                torch.load(cfg.baseline_ckpt)['state_dict']
            )
        )
    if is_search_space:
        model = build_search_space_from_model(model)
        if cfg.phase_name == 'search':
            sample_batch = next(iter(train_loader))

            model.original_model.real_forward = model.original_model.forward
            model.original_model.forward = model.original_model.forward_dummy

            latency_container = initialize_latency('mmac.pthflops', model, (sample_batch['img'],))
            logger.info(f'constant: {latency_container._constant_latency}')
            logger.info(f'operations_latencies: {latency_container._operations_latencies}')

            model.original_model.forward = model.original_model.real_forward

    if cfg.phase_name == 'tune':
        model = build_search_space_from_model(model)
        model.load_state_dict(
            torch.load(
                cfg.ss_checkpoint,
                map_location='cpu',
            )['model'],
            strict=False,
        )
        model = model.get_network_by_indexes(cfg.searched_arch)

    if cfg.resume_from:
        logger.info(
            model.load_state_dict(
                {k: v for k, v in
                 torch.load(
                     cfg.resume_from,
                     map_location="cpu",
                 )["model"].items()},
                strict=False,
            )
        )

    # User has to manually place model to cuda.
    model.cuda()
    # synchronize models weights for multigpu
    synchronize_model_with_checkpoint(model)
    return model


def run_train(
        model_config_path: str,
        experiment_args: Namespace,
        optimizer_constructor: Callable,
        opt_params: Dict[str, Any],
        scheduler: Optional[Scheduler],
        scheduler_params: Optional[Dict[str, Any]],
        warmup_epochs: int,
        batch_size: int,
) -> None:
    """
    Using mmdetection config build model and dataset and start pretrain phase
    """
    # Initialize enot runner params, multigpu params and etc.
    exp_dir = init_exp_dir(experiment_args.rank, experiment_args.exp_dir)
    init_torch(experiment_args.seed, experiment_args.local_rank)
    logger = prepare_log(log_path=exp_dir / 'log_pretrain.txt')
    print('path : ', exp_dir / 'log_pretrain.txt')

    logger.info("Initial preparation ready")
    # For multigpu we use distributed data sampler.
    use_distributed_sampler = dist.is_initialized()

    # Load config and build model
    cfg = Config.fromfile(model_config_path)

    # Now supported only single dataset configs.
    train_dataset = build_dataset(cfg.data.train)
    # is important for valid metrics
    cfg.data.test.test_mode = True
    valid_dataset = build_dataset(cfg.data.test)

    valid_dataloader = CudaDataLoader(
        MMDetDatasetEnumerateWrapper(valid_dataset),
        batch_size=1,
        collate_fn=valid_collate_function,
    )

    sampler_train = DistributedSampler(train_dataset, shuffle=True) if use_distributed_sampler else None
    train_dataloader = CudaDataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=train_collate_function,
        num_workers=experiment_args.jobs,
        sampler=sampler_train,
    )

    # Initialize model
    model = init_model(
        cfg,
        is_search_space=cfg.phase_name in ('search', 'pretrain'),
        train_loader=train_dataloader,
        logger=logger,
    )
    logger.info("Model ready")

    # User must manually place input tensors to cuda.
    logger.info('Dataloaders ready')

    # For pretrain phase we update SearchSpace params.
    if cfg.phase_name in ('train', 'tune'):
        params = model.parameters()
    elif cfg.phase_name == 'pretrain':
        params = model.model_parameters()
    elif cfg.phase_name == 'search':
        params = model.architecture_parameters()
    else:
        raise AttributeError(f'Unknown phase_name: {cfg.phase_name}')

    optimizer = optimizer_constructor(
        params=params,
        **opt_params
    )
    if scheduler:
        scheduler = scheduler(optimizer, T_max=len(train_dataloader) * experiment_args.epochs, **scheduler_params)
    if warmup_epochs > 0:
        scheduler = WarmupScheduler(scheduler, warmup_steps=len(train_dataloader) * warmup_epochs)
    enot_optimizer = build_enot_optimizer(
        phase_name=cfg.phase_name,
        model=model,
        optimizer=optimizer
    )

    logger.info('Train schedule ready')

    torch_save(
        {
            'epoch': 0,
            'model': model.state_dict(),
        },
        os.path.join(exp_dir, 'checkpoint-0.pth'),
    )

    train(
        model=model,
        epochs=experiment_args.epochs,
        train_loader=train_dataloader,
        validation_loader=valid_dataloader,
        enot_optimizer=enot_optimizer,
        scheduler=scheduler,
        metric_function=partial(custom_coco_evaluation, dataset=valid_dataset),
        loss_function=parse_losses,
        validation_forward_wrapper=custom_valid_forward_logic,
        train_forward_wrapper=custom_train_forward_logic,
        exp_dir=exp_dir,
        logger=logger,
        eval_frequency=cfg.eval_freq,
    )


def make_args(
        distributed_args: Namespace,
        config: Config,
):
    return Namespace(
        exp_dir=Path(config.work_dir),
        jobs=config.data['workers_per_gpu'],
        epochs=config.total_epochs,
        rank=distributed_args.rank,
        local_rank=distributed_args.local_rank,
        seed=config.random_seed,
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

    args = make_args(
        command_line_args,
        cfg,
    )

    run_train(
        model_config_path=command_line_args.config_path,
        experiment_args=args,
        optimizer_constructor=getattr(torch.optim, cfg.optimizer['type']),
        opt_params=cfg.optimizer_params,
        scheduler=getattr(torch.optim.lr_scheduler, cfg.scheduler['type']),
        scheduler_params=cfg.scheduler_params,
        warmup_epochs=cfg.warmup_epochs,
        batch_size=cfg.batch_size,
    )
