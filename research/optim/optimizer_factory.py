import torch

from .optimizers import SAM, TopKSAM


def create_sam_optimizer(model, args):
    if args.sam_topk == 0:
        OPTIMIZER_CLASS = SAM
        topk_kw = dict()
    else:
        OPTIMIZER_CLASS = TopKSAM
        topk_kw = dict(topk=args.sam_topk, global_sparsity=args.sam_global_sparsity)

    if args.opt == 'sgd':
        optimizer = OPTIMIZER_CLASS(
            model.parameters(), 
            torch.optim.SGD,
            lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            **topk_kw
        )
    elif args.opt == 'adam':
        if not args.opt_betas:
            args.opt_betas = (0.9, 0.999)
            args.eps = 1e-8
        optimizer = OPTIMIZER_CLASS(
            model.parameters(), 
            torch.optim.Adam,
            rho=args.sam_rho,
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=args.opt_betas,
            eps=args.eps,
            **topk_kw
        )

    elif args.opt == 'adamw':
        if not args.opt_betas:
            args.opt_betas = (0.9, 0.999)
            args.eps = 1e-8
        optimizer = OPTIMIZER_CLASS(
            model.parameters(), 
            torch.optim.AdamW,
            rho=args.sam_rho,
            lr=args.lr, 
            weight_decay=args.weight_decay,
            betas=args.opt_betas,
            eps=args.eps,
            **topk_kw
        )
    else:
        raise NotImplementedError("Unfortunately this kind of optimizer is not supported")
    return optimizer
