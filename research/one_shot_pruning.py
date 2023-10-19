import os
import timm
import torch
import pickle
import argparse
import numpy as np
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.utils import AverageMeter
from timm.data import create_dataset, create_loader

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

from sparseml.pytorch.optim import ScheduledModifierManager


def parse_args():
    parser = argparse.ArgumentParser('One shot pruning of ImageNet pretrained model.', add_help=False)
    # Model
    parser.add_argument(
        '--model', 
        default='deit_small_patch16_224', 
        type=str
    )
    parser.add_argument(
        '--checkpoint_path', 
        default='', 
        type=str, 
        help='Path to model checkpoint'
    )
    # Data
    parser.add_argument(
        '--dataset', 
        default='imagenet', 
        type=str
    )
    parser.add_argument(
        '--data_dir', 
        required=True, 
        type=str
    )
    # Experiment
    parser.add_argument(
        '--seed', 
        default=42, 
        type=int
    )
    # Sparsification params
    parser.add_argument(
        '--sparseml_recipe', 
        required=True, 
        type=str
    )
    parser.add_argument(
        '--sparsities', 
        nargs='+', 
        default=[], 
        required=False, 
        type=float
    )
    # Loader params
    parser.add_argument(
        '-vb', 
        '--val_batch_size', 
        default=128, 
        type=int
    )
    parser.add_argument(
        '--workers', 
        default=4, 
        type=int
    )
    parser.add_argument(
        '--gs_loader', 
        action='store_true',
        help='Whether to create additional loader for grad sampling.'
    )
    parser.add_argument(
        '-gb', 
        '--grad_sampler_batch_size', 
        default=128, 
        type=int
    )
    parser.add_argument(
        '--no_prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--pin_mem', action='store_true', default=False,
                        help='Whether to pin memory')
    # Save arguments
    parser.add_argument(
        '--output_dir', 
        default='./output/one-shot', 
        type=str, 
        help='dir to save results'
    )
    parser.add_argument(
        '--save_model', 
        action='store_true', 
        help='Whether to save pruned model'
    )
    # Logging
    parser.add_argument('--log_wandb', action='store_true')

    args = parser.parse_args()
    return args


def accuracy(logits, labels):
    return (torch.argmax(logits, dim=1) == labels).sum() / len(labels)


@torch.no_grad()
@torch.cuda.amp.autocast()
def val_epoch(
    model : nn.Module, 
    data_loader : DataLoader,
    criterion : nn.Module,
    device : torch.device
):
    model.eval()
    loss_m =  AverageMeter()
    acc_m = AverageMeter()
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        # get model output
        logits = model(images)
        # compute loss
        loss = criterion(logits, labels)         
        # statistics
        loss_m.update(loss.item())
        acc_m.update(accuracy(logits, labels).item())

    return {"loss" : loss_m.avg, "acc" : acc_m.avg}


if __name__ == '__main__':
    # parse args
    args = parse_args()
    # seed all
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert torch.cuda.is_available(), "One needs CUDA to run the code below"
    device = "cuda"
    # set num threads
    torch.set_num_threads(args.workers + 1)
    # init wandb
    if args.log_wandb:
        assert has_wandb
        wandb.init(config=args)

    # Model
    model = timm.create_model(
        args.model, 
        pretrained=True,
        checkpoint_path=args.checkpoint_path
    )
    model = model.to(device)

    ######## 
    # Data #
    ########

    # get transform params
    mean = model.pretrained_cfg['mean']
    std = model.pretrained_cfg['std']
    input_size = model.pretrained_cfg['input_size']
    crop_pct = model.pretrained_cfg['crop_pct']
    interpolation = model.pretrained_cfg['interpolation']

    args.prefetcher = not args.no_prefetcher

    val_dataset = create_dataset(
        args.dataset, 
        root=args.data_dir, 
        split='val', 
        is_training=False,
        batch_size=args.val_batch_size
    )

    val_loader = create_loader(
        val_dataset,
        input_size=input_size,
        batch_size=args.val_batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=interpolation,
        mean=mean,
        std=std,
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=True,
    )

    #####

    loss_fn = F.cross_entropy

    os.makedirs(args.output_dir, exist_ok=True)
    experiment_data = {
        'sparsity': args.sparsities, 'val/acc' : []
    }

    manager_kwargs = {}
    # define for Fisher-based 2nd order pruner
    if args.gs_loader:
        train_dataset = create_dataset(
            args.dataset, 
            root=args.data_dir, 
            split='train', 
            is_training=True,
            batch_size=args.grad_sampler_batch_size
        )

        grad_sampler_loader = create_loader(
            train_dataset,
            input_size=input_size,
            batch_size=args.grad_sampler_batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            interpolation=interpolation,
            mean=mean,
            std=std,
            num_workers=args.workers,
            distributed=False,
            crop_pct=crop_pct,
            pin_memory=args.pin_mem
        )

        def data_loader_builder(device=device, **kwargs):
            while True:
                for input, target in grad_sampler_loader:
                    input, target = input.to(device), target.to(device)
                    yield [input], {}, target

        manager_kwargs['grad_sampler'] = {
            'data_loader_builder' : data_loader_builder, 
            'loss_fn' : loss_fn,
        }

    # evaluate dense model
    val_acc = val_epoch(model, val_loader, loss_fn, device=device)['acc']
    print(f'Test accuracy dense: {val_acc:.3f}')

    for sparsity in args.sparsities:
        print(f'Sparsity {sparsity:.3f}')
        model_sparse = deepcopy(model)
        # create sparseml manager
        manager = ScheduledModifierManager.from_yaml(args.sparseml_recipe)
        # update manager
        manager.modifiers[0].init_sparsity  = sparsity
        manager.modifiers[0].final_sparsity = sparsity
        # apply recipe
        manager.apply(
            model_sparse, 
            **manager_kwargs,
            finalize=True
        )
        # evaluate 
        val_acc = val_epoch(model_sparse, val_loader, loss_fn, device=device)['acc']
        # update experiment data
        experiment_data['val/acc'].append(val_acc)
        print(f'Test accuracy: {val_acc:.3f}')
        if args.log_wandb:
            wandb.log({'sparsity' : sparsity, 'val/acc': val_acc})

        if args.save_model:
            torch.save(
                model_sparse.state_dict(), 
                os.path.join(args.output_dir, f'{args.model}_sparsity={sparsity}.pth')
            )

    with open(f'{args.output_dir}/experiment_data.pkl', 'wb') as fout:
        pickle.dump(experiment_data, fout, protocol=pickle.HIGHEST_PROTOCOL)   

    print('Finished!') 

