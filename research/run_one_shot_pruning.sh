#!/bin/bash

NUM_PROC=$(wc -w <<< $(tr ',' ' ' <<< $CUDA_VISIBLE_DEVICES))
BATCH_SIZE_PER_GPU="<batch size per 1 gpu>"

DATA_DIR="<data dir>"
MODEL="<model>"

# if using W&B for logging
export WANDB_ENTITY="<W&B user name>"
export WANDB_PROJECT="<project name>"
export WANDB_NAME="<run name>"

# to reproduce experiment with convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320
# recipes/one_shot/cap/cap_convnext_bs=128_ng=4096_damp=1e-7.yaml

#  --gs_loader - for OBS/CAP/FastCAP modifiers
#  --grad_sampler_batch_size 128 - for OBS/CAP/FastCAP modifiers

# --save_model - to save model
# --log_wandb - to log to W&B

python one_shot_pruning.py \
    \
    --data_dir ${DATA_DIR} \
    \
    --sparseml_recipe "<sparseml recipe path>" \
    \
    --model ${MODEL} \
    \
    --gs_loader \
    -gb ${GRAD_SAMPLER_BATCH_SIZE} \
    --val_batch_size 256 \
    --workers 8 \
    \
    --output_dir "<output_dir>" \
    \
    --sparsities "<list of sparsitites>"
