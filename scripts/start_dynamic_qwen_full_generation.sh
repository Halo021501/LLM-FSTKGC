#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TAG="${RUN_TAG:-formal_qwen7b_$(date +%Y%m%d_%H%M%S)}"
STATE_DIR="${STATE_DIR:-logs/$RUN_TAG}"
CACHE_DIR="${CACHE_DIR:-cache/standard_rolling_history/qwen2.5-7b-awq}"

if [[ "${CONFIRM_LOCAL_QWEN_GENERATION:-NO}" != "YES" ]]; then
  echo "Refusing the 26,358-query run without CONFIRM_LOCAL_QWEN_GENERATION=YES." >&2
  exit 2
fi
if [[ -e "$STATE_DIR" ]]; then
  echo "State directory already exists: $STATE_DIR" >&2
  exit 2
fi
if [[ -e "$CACHE_DIR/test_s5.jsonl" || -e "$CACHE_DIR/test_s10.jsonl" ]]; then
  echo "A final formal cache already exists under $CACHE_DIR; refusing to overwrite it." >&2
  exit 2
fi

mkdir -p "$STATE_DIR"
nohup setsid "$PYTHON_BIN" scripts/dynamic_local_qwen_pool.py \
  --confirm-full-generation \
  --state-dir "$STATE_DIR" \
  --cache-dir "$CACHE_DIR" \
  --data-dir data/ICEWS14 \
  --initial-gpus 2,3,4,5 \
  --candidate-gpus 0,1,6 \
  --adopt-endpoint 2=http://127.0.0.1:8000/v1 \
  --num-shards 256 \
  --workers-per-gpu 4 \
  --max-tokens 512 \
  --retry-max-tokens 768 \
  --max-retries 1 \
  --task-max-attempts 3 \
  --task-retry-backoff-seconds 30 \
  --request-timeout 360 \
  --poll-seconds 10 \
  --idle-checks 2 \
  --server-retry-cooldown-seconds 120 \
  --additional-min-free-mib 12000 \
  --shared-min-free-mib 9500 \
  >"$STATE_DIR/controller.log" 2>&1 < /dev/null &
controller_pid=$!
echo "$controller_pid" > "$STATE_DIR/launcher.pid"
printf '%s\n' "$STATE_DIR" > logs/dynamic_qwen_pool.latest

echo "Dynamic Qwen formal generation started."
echo "state_dir=$STATE_DIR"
echo "controller_pid=$controller_pid"
echo "status_command=./scripts/check_dynamic_qwen_pool.sh $STATE_DIR"
