#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/ICEWS14}"
CACHE_DIR="${CACHE_DIR:-cache/standard_rolling_history/qwen2.5-7b-awq}"
DEVICE="${DEVICE:-cuda:0}"

for shot in 5 10; do
  for seed in 42 43 44; do
    for mode in off candidate score rationale; do
      output="runs/formal_s${shot}_seed${seed}_${mode}"
      extra=()
      if [[ "$mode" != "off" ]]; then
        extra+=(
          --llm-valid-cache "$CACHE_DIR/valid_s${shot}.jsonl"
          --llm-test-cache "$CACHE_DIR/test_s${shot}.jsonl"
        )
      fi
      "$PYTHON_BIN" train.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$output" \
        --device "$DEVICE" \
        --shot "$shot" \
        --seed "$seed" \
        --epochs 40 \
        --episodes-per-epoch 300 \
        --warmup-epochs 5 \
        --warmup-batch-size 256 \
        --joint-supervised-weight 0.5 \
        --supervised-batch-size 256 \
        --query 16 \
        --history-len 10 \
        --dim 256 \
        --channels 64 \
        --dropout 0.2 \
        --lr 0.0003 \
        --eval-every 5 \
        --eval-batch-size 512 \
        --history-protocol standard_rolling_history \
        --llm-mode "$mode" \
        "${extra[@]}"
    done
  done
done
