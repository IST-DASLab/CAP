import os
import time
import torch
import torch.nn as nn

from logging import Logger
from argparse import Namespace
from collections import OrderedDict
from typing import Optional, Callable
from contextlib import suppress, nullcontext

from torch.optim import Optimizer
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from timm.utils import (
    AverageMeter, 
    reduce_tensor,
    accuracy,
    dispatch_clip_grad
)
from timm.models import model_parameters

from .batchnorm_utils import (
    enable_running_stats, 
    disable_running_stats
)


def train_one_epoch(
    epoch: int, 
    model: nn.Module, 
    loader: DataLoader, 
    optimizer: Optimizer, 
    loss_fn: nn.Module, 
    args: Namespace,
    lr_scheduler = None, 
    output_dir: str = None, 
    amp_autocast = suppress,
    grad_scaler: Optional[GradScaler] = None, 
    model_ema: nn.Module = None, 
    mixup_fn: Callable = None,
    logger: Logger = None
):
    
    # define closure for SAM
    def closure():
        loss = loss_fn(model(input), target)
        loss.backward()
        return loss

    # clip grad fn
    def clip_grad_fn():
        dispatch_clip_grad(
            model_parameters(model, exclude_head='agc' in args.clip_mode),
            value=args.clip_grad, 
            mode=args.clip_mode
        )

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m  = AverageMeter()
    losses_m     = AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher:
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # turn of batch norm stats if needed
        if args.sam:
            enable_running_stats(model)

        with amp_autocast():
            output = model(input)
            loss = loss_fn(output, target)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        if grad_scaler is not None:
            grad_scaler.scale(loss).backward(create_graph=second_order)
            if args.clip_grad is not None:
                grad_scaler.unscale_(optimizer)
                clip_grad_fn()

            if args.sam:
                disable_running_stats(model)
                grad_scaler.step(optimizer, closure)
            else:
                grad_scaler.step(optimizer)
            
            grad_scaler.update()
        else:
            sync_context = model.no_sync() if args.sam else nullcontext
            with sync_context:
                loss.backward(create_graph=second_order)
                if args.clip_grad is not None:
                    clip_grad_fn()

            if args.sam:
                disable_running_stats(model)
                optimizer.step(closure)
            else:
                optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        lr = 0.0
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if args.local_rank == 0:
                logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m
                    )
                )

                if args.save_images and output_dir:
                    save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True
                    )

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def validate(
    model: nn.Module, 
    loader: DataLoader, 
    loss_fn: nn.Module, 
    args: Namespace, 
    amp_autocast=suppress, 
    log_suffix='',
    logger: Logger = None
):
    batch_time_m = AverageMeter()
    losses_m     = AverageMeter()
    top1_m       = AverageMeter()

    model.eval()
    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0: target.size(0): reduce_factor]

            loss = loss_fn(output, target)
            top1 = accuracy(output, target, topk=(1,))[0]

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                top1 = reduce_tensor(top1, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(top1.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m
                    )
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg)])

    return metrics
