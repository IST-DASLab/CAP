import torch
import numpy as np

from sklearn.model_selection import train_test_split


__all__ = ['load_calibration_images']


def load_calibration_images(train_dataset, args, _logger):
    if args.local_rank == 0:
        _logger.info(f"Collecting {args.num_calibration_images} for AdaPrune")
    train_ids_all = range(len(train_dataset))
    train_labels_all = np.load(args.path_to_labels)
    train_ids, *dummy =  train_test_split(
        train_ids_all, train_labels_all, stratify=train_labels_all, train_size=args.num_calibration_images)
    calibration_images = torch.stack(
        [torch.tensor(train_dataset[i][0], device=args.device, dtype=torch.float32) for i in train_ids], dim=0)
    return calibration_images
    