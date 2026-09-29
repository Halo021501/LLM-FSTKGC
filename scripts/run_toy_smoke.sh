#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON_BIN:-python}" train.py \
  --data-dir data/toy \
  --output-dir runs/toy_smoke \
  --device cpu \
  --warmup-epochs 1 \
  --warmup-batches-per-epoch 2 \
  --warmup-batch-size 4 \
  --joint-supervised-weight 0.5 \
  --supervised-batch-size 4 \
  --epochs 1 \
  --episodes-per-epoch 3 \
  --shot 2 \
  --query 2 \
  --history-len 4 \
  --dim 32 \
  --channels 8 \
  --eval-limit 4
