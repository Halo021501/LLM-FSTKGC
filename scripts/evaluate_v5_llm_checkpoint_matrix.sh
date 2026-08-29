#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/ICEWS14}"
CACHE_DIR="${CACHE_DIR:-cache/standard_rolling_history/qwen2.5-7b-awq}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/alterego_v5}"

for shot in 5 10; do
  for seed in 42 43 44; do
    checkpoint="$CHECKPOINT_ROOT/main_s${shot}_seed${seed}/best.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing validation-selected v5 checkpoint: $checkpoint" >&2
      exit 2
    fi
    for mode in off candidate score rationale; do
      extra=()
      if [[ "$mode" != "off" ]]; then
        extra+=(
          --llm-test-cache "$CACHE_DIR/test_s${shot}.jsonl"
        )
      fi
      "$PYTHON_BIN" train.py \
        --data-dir "$DATA_DIR" \
        --output-dir "runs/frozen_v5_s${shot}_seed${seed}_${mode}" \
        --device "$DEVICE" \
        --seed "$seed" \
        --shot "$shot" \
        --query 16 \
        --history-len 10 \
        --dim 256 \
        --channels 64 \
        --dropout 0.2 \
        --epochs 0 \
        --episodes-per-epoch 0 \
        --warmup-epochs 0 \
        --eval-batch-size 512 \
        --history-protocol standard_rolling_history \
        --llm-mode "$mode" \
        --init-from-v5 "$checkpoint" \
        "${extra[@]}"
    done
  done
done
