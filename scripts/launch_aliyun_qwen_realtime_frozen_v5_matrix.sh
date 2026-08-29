#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${1:-$ROOT/runs/aliyun_qwen_realtime_qwen_flash_20260809}"
GPUS=(${GPUS:-4})
SHOTS=(${SHOTS:-1 3 5 10})
SEEDS=(${SEEDS:-42 43 44})
MODES=(${MODES:-candidate score rationale})
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-5200}"
MIN_HOST_AVAILABLE_MB="${MIN_HOST_AVAILABLE_MB:-2500}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STATE_DIR="$RUN_ROOT/task_state"
CLAIM_DIR="$STATE_DIR/claims"
QUEUE_LOCK="$STATE_DIR/queue.lock"
FATAL_FILE="$RUN_ROOT/PIPELINE_FAILED"

if (( ${#GPUS[@]} == 0 )); then
  echo "at least one GPU is required" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/frozen" "$RUN_ROOT/gpu_monitor" "$CLAIM_DIR"
printf '%s\n' "$$" > "$STATE_DIR/controller.$$.pid"

if ! (cd "$ROOT" && sha256sum -c --status SOURCE_MANIFEST.sha256); then
  echo "source manifest verification failed" >&2
  exit 4
fi

TASKS=()
for shot in "${SHOTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for mode in "${MODES[@]}"; do
      TASKS+=("${shot}:${seed}:${mode}")
    done
  done
done

if (( ${#TASKS[@]} == 0 )); then
  echo "empty task matrix" >&2
  exit 2
fi

{
  echo "provider=aliyun_qwen_realtime"
  echo "model=qwen-flash"
  echo "history_protocol=standard_rolling_history"
  echo "shots=${SHOTS[*]}"
  echo "seeds=${SEEDS[*]}"
  echo "modes=${MODES[*]}"
  echo "gpus=${GPUS[*]}"
  echo "task_count=${#TASKS[@]}"
  echo "off_baseline=reuse_validation_selected_v5_runs"
  echo "llm_api_calls=false"
  echo "source_manifest_sha256=$(sha256sum "$ROOT/SOURCE_MANIFEST.sha256" | awk '{print $1}')"
} > "$RUN_ROOT/experiment_config.txt"
cp "$ROOT/SOURCE_MANIFEST.sha256" "$RUN_ROOT/SOURCE_MANIFEST.launch.sha256"

task_name() {
  local spec="$1" shot seed mode
  IFS=: read -r shot seed mode <<< "$spec"
  printf 'frozen_s%s_seed%s_%s' "$shot" "$seed" "$mode"
}

is_complete() {
  local name="$1" out="$RUN_ROOT/frozen/$name"
  [[ -s "$out/metrics.json" && -s "$out/run_meta.json" && -s "$out/exit_code.txt" ]] \
    && [[ "$(tr -dc '0-9' < "$out/exit_code.txt")" == "0" ]]
}

claim_next() {
  local gpu="$1" claim_file="$2" spec name
  while ! mkdir "$QUEUE_LOCK" 2>/dev/null; do sleep 0.2; done
  for spec in "${TASKS[@]}"; do
    name="$(task_name "$spec")"
    if is_complete "$name" || [[ -e "$CLAIM_DIR/$name" ]]; then
      continue
    fi
    {
      echo "controller_pid=$$"
      echo "worker_pid=$BASHPID"
      echo "gpu=$gpu"
      echo "claimed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    } > "$CLAIM_DIR/$name"
    printf '%s\n' "$spec" > "$claim_file"
    rmdir "$QUEUE_LOCK"
    return 0
  done
  rmdir "$QUEUE_LOCK"
  return 1
}

host_available_mb() {
  awk '/MemAvailable:/ {printf "%d\n", $2 / 1024}' /proc/meminfo
}

gpu_free_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | tr -dc '0-9'
}

wait_for_capacity() {
  local gpu="$1" host_mb gpu_mb
  while [[ ! -f "$FATAL_FILE" ]]; do
    host_mb="$(host_available_mb)"
    gpu_mb="$(gpu_free_mb "$gpu")"
    if [[ -n "$gpu_mb" ]] && (( host_mb >= MIN_HOST_AVAILABLE_MB && gpu_mb >= MIN_GPU_FREE_MB )); then
      return 0
    fi
    echo "[$(date '+%F %T')] WAIT_CAPACITY gpu=$gpu host_available_mb=$host_mb gpu_free_mb=${gpu_mb:-unknown}" \
      | tee -a "$RUN_ROOT/matrix.status"
    sleep "$POLL_SECONDS"
  done
  return 1
}

run_worker() {
  local gpu="$1" claim_file spec shot seed mode name status
  claim_file="$STATE_DIR/claim_gpu${gpu}_$BASHPID.txt"
  echo "[$(date '+%F %T')] WORKER_START gpu=$gpu pid=$BASHPID" | tee -a "$RUN_ROOT/matrix.status"
  while [[ ! -f "$FATAL_FILE" ]]; do
    if ! claim_next "$gpu" "$claim_file"; then
      break
    fi
    spec="$(<"$claim_file")"
    IFS=: read -r shot seed mode <<< "$spec"
    name="$(task_name "$spec")"
    if ! wait_for_capacity "$gpu"; then
      break
    fi
    set +e
    "$ROOT/scripts/run_aliyun_qwen_realtime_frozen_task.sh" \
      "$shot" "$seed" "$mode" "$gpu" "$RUN_ROOT"
    status="$?"
    set -e
    if (( status != 0 )); then
      {
        echo "task=$name"
        echo "gpu=$gpu"
        echo "exit_code=$status"
        echo "failed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      } > "$FATAL_FILE"
      break
    fi
  done
  echo "[$(date '+%F %T')] WORKER_DONE gpu=$gpu pid=$BASHPID" | tee -a "$RUN_ROOT/matrix.status"
}

echo "[$(date '+%F %T')] MATRIX_START tasks=${#TASKS[@]} gpus=${GPUS[*]}" | tee -a "$RUN_ROOT/matrix.status"
WORKER_PIDS=()
for gpu in "${GPUS[@]}"; do
  run_worker "$gpu" &
  WORKER_PIDS+=("$!")
  sleep 3
done
printf '%s\n' "${WORKER_PIDS[@]}" > "$STATE_DIR/worker.pids"

STATUS=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then
    STATUS=1
  fi
done

COMPLETE=0
for spec in "${TASKS[@]}"; do
  if is_complete "$(task_name "$spec")"; then
    COMPLETE=$((COMPLETE + 1))
  fi
done
echo "[$(date '+%F %T')] MATRIX_END complete=$COMPLETE total=${#TASKS[@]} status=$STATUS" \
  | tee -a "$RUN_ROOT/matrix.status"

if [[ -f "$FATAL_FILE" ]] || (( STATUS != 0 || COMPLETE != ${#TASKS[@]} )); then
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/PIPELINE_DONE"
