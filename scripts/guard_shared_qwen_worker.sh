#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd -P)"

STATE_DIR="${1:?usage: guard_shared_qwen_worker.sh STATE_DIR GPU_ID [MIN_AVAILABLE_MIB]}"
GPU_ID="${2:?usage: guard_shared_qwen_worker.sh STATE_DIR GPU_ID [MIN_AVAILABLE_MIB]}"
MIN_AVAILABLE_MIB="${3:-2048}"
CHECK_SECONDS="${QWEN_GUARD_CHECK_SECONDS:-20}"
CONSECUTIVE_LIMIT="${QWEN_GUARD_CONSECUTIVE_LIMIT:-3}"
MIN_GPU_FREE_MIB="${QWEN_GUARD_MIN_GPU_FREE_MIB:-1024}"
STATE_DIR="$(realpath -m "$STATE_DIR")"
SIDECAR_PID_FILE="$STATE_DIR/servers/gpu${GPU_ID}_sidecar.pid"
SERVER_STATE_DIR="$STATE_DIR/servers/gpu${GPU_ID}"
SERVER_PID_FILE="$SERVER_STATE_DIR/server.pid"
LOG_FILE="$STATE_DIR/servers/gpu${GPU_ID}_guard.log"

if [[ "$STATE_DIR" != "$PROJECT_ROOT"/* ]]; then
  echo "STATE_DIR must be inside this project." >&2
  exit 2
fi
if ! [[ "$GPU_ID" =~ ^[0-9]+$ && "$MIN_AVAILABLE_MIB" =~ ^[0-9]+$ \
  && "$CHECK_SECONDS" =~ ^[0-9]+$ && "$CONSECUTIVE_LIMIT" =~ ^[0-9]+$ \
  && "$MIN_GPU_FREE_MIB" =~ ^[0-9]+$ ]]; then
  echo "GPU id, resource floors, and guard intervals must be non-negative integers." >&2
  exit 2
fi
if (( CHECK_SECONDS < 1 || CONSECUTIVE_LIMIT < 1 )); then
  echo "Guard check interval and consecutive limit must be positive." >&2
  exit 2
fi

umask 077
mkdir -p "$(dirname "$LOG_FILE")"

timestamp() {
  date -u +'%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '%s %s\n' "$(timestamp)" "$*" >>"$LOG_FILE"
}

read_pid() {
  local file="$1"
  local value=""
  [[ -s "$file" ]] || return 1
  value="$(tr -d '[:space:]' <"$file")"
  [[ "$value" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$value"
}

owned_process() {
  local pid="$1"
  local required="$2"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  [[ "$(stat -c '%u' "/proc/$pid" 2>/dev/null)" == "$(id -u)" ]] || return 1
  [[ "$(readlink -f "/proc/$pid/cwd" 2>/dev/null)" == "$PROJECT_ROOT" ]] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq -- "$required"
}

stop_owned_stack() {
  local reason="$1"
  local sidecar_pid=""
  local server_pid=""
  local child_pid=""
  log "guard_triggered reason=$reason"

  sidecar_pid="$(read_pid "$SIDECAR_PID_FILE" 2>/dev/null || true)"
  if owned_process "$sidecar_pid" "attach_shared_qwen_worker.py"; then
    while read -r child_pid; do
      child_pid="${child_pid//[[:space:]]/}"
      if owned_process "$child_pid" "stlp_generate_candidates.py"; then
        kill -TERM "$child_pid" 2>/dev/null || true
        log "generator_term_sent pid=$child_pid"
      fi
    done < <(ps -o pid= --ppid "$sidecar_pid" 2>/dev/null || true)
    kill -TERM "$sidecar_pid" 2>/dev/null || true
    log "sidecar_term_sent pid=$sidecar_pid"
  fi

  for _ in $(seq 1 20); do
    if ! owned_process "$sidecar_pid" "attach_shared_qwen_worker.py"; then
      break
    fi
    sleep 1
  done

  server_pid="$(read_pid "$SERVER_PID_FILE" 2>/dev/null || true)"
  if owned_process "$server_pid" "serve_local_qwen_loopback.py"; then
    env LOCAL_QWEN_ENV_FILE=/dev/null LOCAL_QWEN_STATE_DIR="$SERVER_STATE_DIR" \
      bash scripts/stop_local_qwen_server.sh >>"$LOG_FILE" 2>&1 || true
  fi
  log "guard_exit"
}

low_count=0
log "guard_started gpu=$GPU_ID min_available_mib=$MIN_AVAILABLE_MIB min_gpu_free_mib=$MIN_GPU_FREE_MIB consecutive_limit=$CONSECUTIVE_LIMIT"
while true; do
  sidecar_pid="$(read_pid "$SIDECAR_PID_FILE" 2>/dev/null || true)"
  server_pid="$(read_pid "$SERVER_PID_FILE" 2>/dev/null || true)"
  if ! owned_process "$sidecar_pid" "attach_shared_qwen_worker.py"; then
    if owned_process "$server_pid" "serve_local_qwen_loopback.py"; then
      stop_owned_stack "sidecar_not_running"
    else
      log "stack_finished"
    fi
    exit 0
  fi
  if ! owned_process "$server_pid" "serve_local_qwen_loopback.py"; then
    stop_owned_stack "server_not_running"
    exit 1
  fi

  available_mib="$(awk '/MemAvailable:/ {printf "%d", $2 / 1024}' /proc/meminfo)"
  gpu_free_mib="$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -dc '0-9')"
  if (( available_mib < MIN_AVAILABLE_MIB || gpu_free_mib < MIN_GPU_FREE_MIB )); then
    low_count=$((low_count + 1))
    log "pressure count=$low_count available_mib=$available_mib gpu_free_mib=$gpu_free_mib"
  else
    if (( low_count > 0 )); then
      log "pressure_cleared available_mib=$available_mib gpu_free_mib=$gpu_free_mib"
    fi
    low_count=0
  fi
  if (( low_count >= CONSECUTIVE_LIMIT )); then
    stop_owned_stack "resource_pressure"
    exit 2
  fi
  sleep "$CHECK_SECONDS"
done
