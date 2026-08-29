#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd -P)"
QWEN_ENV_FILE="${LOCAL_QWEN_ENV_FILE:-.env.qwen_local}"
if [[ -f "$QWEN_ENV_FILE" ]]; then
  set -a
  source "$QWEN_ENV_FILE"
  set +a
fi

QWEN_PYTHON="${QWEN_PYTHON:-python}"
SERVE_ENTRYPOINT="${LOCAL_QWEN_SERVE_ENTRYPOINT:-scripts/serve_local_qwen_loopback.py}"
MODEL_DIR="${LOCAL_QWEN_MODEL_DIR:-$PROJECT_ROOT/models/Qwen2.5-7B-Instruct-AWQ}"
SERVED_MODEL_NAME="${LOCAL_QWEN_MODEL:-Qwen2.5-7B-Instruct-AWQ}"
GPU_ID="${LOCAL_QWEN_GPU_ID:-0}"
PORT="${LOCAL_QWEN_PORT:-8000}"
MAX_MODEL_LEN="${LOCAL_QWEN_MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${LOCAL_QWEN_GPU_MEMORY_UTILIZATION:-0.70}"
MAX_NUM_SEQS="${LOCAL_QWEN_MAX_NUM_SEQS:-16}"
MIN_FREE_MIB="${LOCAL_QWEN_MIN_FREE_MIB:-9500}"
ENFORCE_EAGER="${LOCAL_QWEN_ENFORCE_EAGER:-NO}"
DISABLE_FRONTEND_MP="${LOCAL_QWEN_DISABLE_FRONTEND_MULTIPROCESSING:-YES}"
GUIDED_DECODING_BACKEND="${LOCAL_QWEN_GUIDED_DECODING_BACKEND:-lm-format-enforcer}"
QUANTIZATION="${LOCAL_QWEN_QUANTIZATION:-awq_marlin}"
STATE_DIR="${LOCAL_QWEN_STATE_DIR:-logs/local_qwen}"
PID_FILE="$STATE_DIR/server.pid"
LOG_FILE="$STATE_DIR/server.log"
START_LOCK_FILE="$STATE_DIR/start.lock"
START_LOCK_TIMEOUT_SECONDS="${LOCAL_QWEN_START_LOCK_TIMEOUT_SECONDS:-300}"

authoritative_state=""
state_parent="$(dirname "$STATE_DIR")"
if [[ "$(basename "$state_parent")" == "servers" ]]; then
  authoritative_state="$(dirname "$state_parent")"
fi

check_reboot_inhibit() {
  if [[ -n "$authoritative_state" ]] && [[ -e "$authoritative_state/PRE_REBOOT_CHECKPOINT.lock" ]]; then
    echo "Pre-reboot checkpoint inhibits local-Qwen startup: $authoritative_state/PRE_REBOOT_CHECKPOINT.lock" >&2
    exit 7
  fi
}

check_reboot_inhibit

process_is_local_qwen() {
  local candidate_pid="$1"
  local command_line
  [[ -r "/proc/$candidate_pid/cmdline" ]] || return 1
  [[ "$(readlink -f "/proc/$candidate_pid/cwd" 2>/dev/null)" == "$PROJECT_ROOT" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/$candidate_pid/cmdline")"
  [[ "$command_line" == *"$(basename "$SERVE_ENTRYPOINT")"* ]]
}

if [[ ! -x "$QWEN_PYTHON" ]]; then
  echo "Missing qwen_local Python: $QWEN_PYTHON" >&2
  exit 2
fi
if [[ ! -s "$SERVE_ENTRYPOINT" ]]; then
  echo "Missing loopback server entrypoint: $SERVE_ENTRYPOINT" >&2
  exit 2
fi
if [[ ! -s "$MODEL_DIR/config.json" ]]; then
  echo "Missing local model at $MODEL_DIR; run scripts/download_local_qwen_model.sh first." >&2
  exit 2
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]]; then
  echo "LOCAL_QWEN_PORT must be numeric." >&2
  exit 2
fi

umask 077
mkdir -p "$STATE_DIR"
if ! [[ "$START_LOCK_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "LOCAL_QWEN_START_LOCK_TIMEOUT_SECONDS must be a non-negative integer." >&2
  exit 2
fi
# The dynamic controller and the opportunistic shared-card supervisor both use
# this launcher and the same per-GPU state directory.  Serializing the complete
# PID/port check plus startup prevents them from racing into two vLLM servers.
exec {start_lock_fd}>"$START_LOCK_FILE"
if ! flock -w "$START_LOCK_TIMEOUT_SECONDS" "$start_lock_fd"; then
  echo "Timed out waiting for the local-Qwen startup lock: $START_LOCK_FILE" >&2
  exit 6
fi
# The inhibit can appear while this launcher waits as long as five minutes on
# the per-GPU startup lock.  Recheck only after owning that lock.
check_reboot_inhibit
if [[ -s "$PID_FILE" ]]; then
  old_pid="$(tr -dc '0-9' < "$PID_FILE")"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null && process_is_local_qwen "$old_pid"; then
    echo "Local Qwen server already running with PID $old_pid" >&2
    exit 2
  fi
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "PID file points to a live non-Qwen process; refusing to overwrite it: $old_pid" >&2
    exit 3
  fi
  rm -f "$PID_FILE"
fi

memory_csv="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.used,memory.free --format=csv,noheader,nounits)"
used_mib="$(cut -d, -f1 <<<"$memory_csv" | tr -dc '0-9')"
free_mib="$(cut -d, -f2 <<<"$memory_csv" | tr -dc '0-9')"
if [[ -z "$used_mib" || -z "$free_mib" ]]; then
  echo "Unable to query GPU $GPU_ID" >&2
  exit 2
fi
if (( used_mib > 512 )) && [[ "${ALLOW_SHARED_GPU:-NO}" != "YES" ]]; then
  echo "GPU $GPU_ID is not idle (${used_mib} MiB already used). Refusing to interfere with an existing experiment." >&2
  exit 3
fi
if (( used_mib > 512 && free_mib < MIN_FREE_MIB )); then
  echo "GPU $GPU_ID has only ${free_mib} MiB free; shared mode requires at least ${MIN_FREE_MIB} MiB." >&2
  exit 3
fi

eager_args=()
if [[ "$ENFORCE_EAGER" == "YES" ]]; then
  eager_args+=(--enforce-eager)
fi
frontend_args=()
if [[ "$DISABLE_FRONTEND_MP" == "YES" ]]; then
  frontend_args+=(--disable-frontend-multiprocessing)
fi

if ss -ltn "sport = :$PORT" | grep -q LISTEN; then
  echo "Port $PORT is already in use." >&2
  exit 3
fi

# Keep the check adjacent to spawn as well; the Python entrypoint repeats the
# same authoritative check before importing/loading vLLM.
check_reboot_inhibit

# Close the startup-lock descriptor in the server child.  The launcher keeps
# the lock through the readiness check, while the long-lived Python process
# does not accidentally hold it for its entire lifetime.
(
  exec {start_lock_fd}>&-
  exec nohup setsid env \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LOCAL_QWEN_STATE_DIR="$STATE_DIR" \
    LOCAL_QWEN_RDZV_DIR="$STATE_DIR/rendezvous" \
    GLOO_SOCKET_IFNAME=lo \
    NCCL_SOCKET_IFNAME=lo \
    "$QWEN_PYTHON" "$SERVE_ENTRYPOINT" "$MODEL_DIR" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --host 127.0.0.1 \
      --port "$PORT" \
      --quantization "$QUANTIZATION" \
      --dtype half \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --guided-decoding-backend "$GUIDED_DECODING_BACKEND" \
      "${eager_args[@]}" \
      "${frontend_args[@]}"
) >"$LOG_FILE" 2>&1 < /dev/null &
server_pid=$!
echo "$server_pid" > "$PID_FILE"

echo "Starting local Qwen on physical GPU $GPU_ID (PID $server_pid; ${free_mib} MiB free before launch)."
for _ in $(seq 1 120); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "Local Qwen exited during startup. Last log lines:" >&2
    tail -80 "$LOG_FILE" >&2
    exit 4
  fi
  if curl --fail --silent --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "Local Qwen ready: http://127.0.0.1:${PORT}/v1"
    exit 0
  fi
  sleep 2
done

echo "Startup is still in progress after 240 seconds; inspect $LOG_FILE and do not generate caches yet." >&2
exit 5
