#!/bin/bash

NUM_PROC=$(wc -w <<< $(tr ',' ' ' <<< $CUDA_VISIBLE_DEVICES))
MASTER_PORT="<master port>" 
BATCH_SIZE_PER_GPU="<batch size per 1 gpu>"

DATA_DIR="<data dir>"
MODEL="<model>"
EXP="<experiment name>"

# if using W&B for logging
export WANDB_ENTITY="<W&B user name>"
export WANDB_PROJECT="<project name>"
export WANDB_NAME="<run name>"

#  --gs-loader - for OBS/CAP/FastCAP modifiers
#  --gb 128 - for OBS/CAP/FastCAP modifiers

python -m torch.distributed.launch \
    --nproc_per_node=${NUM_PROC} \
    --master_port=${MASTER_PORT} \
    sparse_training.py \
    \
    ${DATA_DIR} \
    \
    --config "<config name>" \
    --sparseml-recipe "<sparseml recipe path>" \
    \
    --dataset "<dataset>" \
    \
    --model ${MODEL} \
    --pretrained \
    \
    --experiment ${EXP} \
    \
    -b ${BATCH_SIZE_PER_GPU} \
    -vb 500 \
    --workers 16 \
    \
    --log-sparsity \
    --log-param-histogram \
    \
    --amp \
    \
    --checkpoint-freq "<save checkpoint interval>" \
    --save-last \
    \
    --timeout 18000 \
    --output "<output _dir>"
