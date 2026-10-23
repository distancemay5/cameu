#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the prepared CUB directory}"
: "${RECOGNIZER_CKPT:?Set RECOGNIZER_CKPT to the CUB ResNet-50 checkpoint}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun --nproc_per_node="${NPROC_PER_NODE}" train.py \
  --data-root "${DATA_ROOT}" \
  --recognizer-config configs/resnet50_cub.py \
  --recognizer-checkpoint "${RECOGNIZER_CKPT}" \
  --output outputs/cameu \
  --epochs 50 \
  --batch-size 2 \
  --workers 4 \
  --seed 20260725
