import os
import sys
import torch
import logging
import torch.distributed as dist

from argparse import Namespace
from datetime import timedelta


__all__ = ['init_distributed']


def init_distributed(args: Namespace, logger: logging.Logger):
    # set defaults
    args.distributed = False
    args.device = 'cuda:0'
    args.rank = 0
    args.local_rank = 0
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        # mark if training in distributed mode
        args.distributed = args.world_size > 1
        # set device
        args.device = 'cuda:%d' % args.rank
        torch.cuda.set_device(args.rank)
         # init process group
        if args.distributed:
            dist.init_process_group(
                backend=args.dist_backend, 
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank,
                timeout=timedelta(seconds=args.timeout)
            )
            args.rank = torch.distributed.get_rank()
        # set logging level
        logger.setLevel(level=logging.INFO if args.rank == 0 else logging.WARNING)
        logger.info(f'Training on {args.world_size} GPU')
    elif torch.cuda.is_available():
        logger.setLevel(level=logging.INFO)
        logger.info('Training with a single process on 1 GPUs.')
    else:
        logger.info('No GPU, no party')
        sys.exit(1)
