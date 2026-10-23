#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the prepared CUB directory}"
: "${RECOGNIZER_CKPT:?Set RECOGNIZER_CKPT to the CUB ResNet-50 checkpoint}"

GPU_ID="${GPU_ID:-0}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python evaluate.py \
  --data-root "${DATA_ROOT}" \
  --recognizer-config configs/resnet50_cub.py \
  --recognizer-checkpoint "${RECOGNIZER_CKPT}" \
  --checkpoint outputs/cameu/epoch_50.pth \
  --output outputs/cameu/result.json \
  --batch-size 8 \
  --workers 4
