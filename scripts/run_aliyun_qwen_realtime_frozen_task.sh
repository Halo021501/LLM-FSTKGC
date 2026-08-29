#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-$ROOT/data/ICEWS14}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache/standard_rolling_history/aliyun_qwen_realtime/qwen-flash/standard}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$ROOT/checkpoints/alterego_v5}"

if (( $# != 5 )); then
  echo "usage: $0 SHOT SEED MODE PHYSICAL_GPU RUN_ROOT" >&2
  exit 2
fi

SHOT="$1"
SEED="$2"
MODE="$3"
GPU="$4"
RUN_ROOT="$5"
TASK="frozen_s${SHOT}_seed${SEED}_${MODE}"
OUT_DIR="$RUN_ROOT/frozen/$TASK"
LOG_DIR="$RUN_ROOT/logs"
STATE_DIR="$RUN_ROOT/task_state"
LOG_FILE="$LOG_DIR/${TASK}.log"
LOCK_FILE="$STATE_DIR/locks/${TASK}.lock"
EXIT_FILE="$OUT_DIR/exit_code.txt"
CHECKPOINT="$CHECKPOINT_ROOT/main_s${SHOT}_seed${SEED}/best.pt"
CACHE="$CACHE_DIR/test_s${SHOT}.jsonl"

case "$SHOT" in 1|3|5|10) ;; *) echo "invalid shot: $SHOT" >&2; exit 2 ;; esac
case "$SEED" in 42|43|44) ;; *) echo "invalid seed: $SEED" >&2; exit 2 ;; esac
case "$MODE" in off|candidate|score|rationale) ;; *) echo "invalid mode: $MODE" >&2; exit 2 ;; esac
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $GPU" >&2; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "missing python: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 2; }
if [[ "$MODE" != "off" ]]; then
  [[ -f "$CACHE" && -f "$CACHE.meta.json" ]] || { echo "missing formal cache: $CACHE" >&2; exit 2; }
fi

mkdir -p "$LOG_DIR" "$STATE_DIR/locks" "$RUN_ROOT/failed_artifacts"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "task already locked: $TASK" >&2
  exit 75
fi

is_complete() {
  [[ -s "$OUT_DIR/metrics.json" && -s "$OUT_DIR/run_meta.json" && -s "$EXIT_FILE" ]] || return 1
  [[ "$(tr -dc '0-9' < "$EXIT_FILE")" == "0" ]] || return 1
  "$PYTHON_BIN" - "$OUT_DIR" "$MODE" "$CHECKPOINT" "$CACHE" <<'PY'
import json
import os
import sys

out_dir, mode, checkpoint, cache = sys.argv[1:]
with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as handle:
    metrics = json.load(handle)
with open(os.path.join(out_dir, "run_meta.json"), encoding="utf-8") as handle:
    meta = json.load(handle)
assert meta["version"] == "1.7.0alterego_v5_llm"
assert meta["model_config"]["llm_mode"] == mode
assert os.path.realpath(meta["checkpoint_initialization"]["path"]) == os.path.realpath(checkpoint)
assert float(metrics["test_tie_avg_mrr"]) >= 0.0
assert float(metrics["test_tie_avg_hits10"]) >= 0.0
if mode != "off":
    llm = meta["llm"]
    assert llm["frozen_parent_evaluation"] is True
    assert os.path.realpath(llm["test_cache"]["path"]) == os.path.realpath(cache)
    assert float(llm["test_cache_coverage"]["cache_hit_rate"]) == 1.0
    assert float(metrics["test_llm_cache_hit_rate"]) == 1.0
PY
}

if is_complete; then
  echo "[$(date '+%F %T')] SKIP task=$TASK complete=true"
  exit 0
fi

if [[ -e "$OUT_DIR" ]]; then
  FAILED_DIR="$RUN_ROOT/failed_artifacts/${TASK}.$(date -u +%Y%m%dT%H%M%SZ).$$"
  mv "$OUT_DIR" "$FAILED_DIR"
  echo "[$(date '+%F %T')] PRESERVE_INCOMPLETE task=$TASK path=$FAILED_DIR"
fi
mkdir -p "$OUT_DIR" "$RUN_ROOT/gpu_monitor"

COMMAND=(
  "$PYTHON_BIN" "$ROOT/train.py"
  --data-dir "$DATA_DIR"
  --output-dir "$OUT_DIR"
  --device cuda:0
  --resource-log "$RUN_ROOT/gpu_monitor/${TASK}_allocator.csv"
  --seed "$SEED"
  --shot "$SHOT"
  --query 16
  --history-len 10
  --dim 256
  --channels 64
  --dropout 0.2
  --epochs 0
  --episodes-per-epoch 0
  --warmup-epochs 0
  --eval-batch-size 512
  --history-protocol standard_rolling_history
  --llm-mode "$MODE"
  --init-from-v5 "$CHECKPOINT"
)
if [[ "$MODE" != "off" ]]; then
  COMMAND+=(--llm-test-cache "$CACHE")
fi

{
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'physical_gpu=%s\n' "$GPU"
  printf 'source_manifest_sha256=%s\n' "$(sha256sum "$ROOT/SOURCE_MANIFEST.sha256" | awk '{print $1}')"
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} > "$OUT_DIR/command.txt"

echo "[$(date '+%F %T')] START task=$TASK gpu=$GPU" | tee -a "$RUN_ROOT/matrix.status"
set +e
CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  "${COMMAND[@]}" > "$LOG_FILE" 2>&1
STATUS="$?"
set -e
printf '%s\n' "$STATUS" > "$EXIT_FILE.tmp"
mv "$EXIT_FILE.tmp" "$EXIT_FILE"
echo "[$(date '+%F %T')] END task=$TASK gpu=$GPU status=$STATUS" | tee -a "$RUN_ROOT/matrix.status"

if (( STATUS != 0 )); then
  exit "$STATUS"
fi
if ! is_complete; then
  echo "post-run contract validation failed: $TASK" >&2
  exit 3
fi
