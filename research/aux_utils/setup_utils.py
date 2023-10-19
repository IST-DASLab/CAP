import torch
import torch.nn as nn

from logging import Logger
from argparse import Namespace
from contextlib import suppress
from torch.cuda.amp import GradScaler

from timm.loss import (
    JsdCrossEntropy, 
    BinaryCrossEntropy, 
    SoftTargetCrossEntropy,
    LabelSmoothingCrossEntropy
)

from timm.data import (
    Mixup, 
    FastCollateMixup, 
)

__all__ = [
    "setup_amp",
    "setup_mixup",
    "setup_loss_fn",
]


def setup_amp(args: Namespace, logger: Logger):
    '''
    setup automatic mixed-precision (AMP) loss scaling and op casting
    '''
    # set defaults
    grad_scaler = None
    amp_autocast = suppress 
    # init grad_scaler
    if args.amp:
        amp_autocast = torch.cuda.amp.autocast
        grad_scaler = GradScaler()
        if args.local_rank == 0:
            logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            logger.info('AMP not enabled. Training in float32.')
    return grad_scaler, amp_autocast


def setup_mixup(args: Namespace):
    '''
    setup mixup / cutmix
    '''
    # init defaults 
    mixup_fn = None
    collate_fn = None
    args.mixup_active = False
    mixup_active = args.mixup > 0 or args.cutmix > 0 or args.cutmix_minmax is not None
    if mixup_active:
        args.mixup_active = True
        mixup_args = dict(
            mixup_alpha=args.mixup, 
            cutmix_alpha=args.cutmix, 
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, 
            switch_prob=args.mixup_switch_prob, 
            mode=args.mixup_mode,
            label_smoothing=args.smoothing, 
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not args.aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)
    return mixup_fn, collate_fn



def setup_loss_fn(args: Namespace):
    # setup loss function
    if args.jsd_loss:
        assert args.aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=args.aug_splits, smoothing=args.smoothing)
    elif args.mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(args.device)
    valid_loss_fn = nn.CrossEntropyLoss().to(args.device)

    return train_loss_fn, valid_loss_fn
