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
PORT="${LOCAL_QWEN_PORT:-8000}"
STATE_DIR="${LOCAL_QWEN_STATE_DIR:-logs/local_qwen}"
PID_FILE="$STATE_DIR/server.pid"
SERVE_ENTRYPOINT="${LOCAL_QWEN_SERVE_ENTRYPOINT:-scripts/serve_local_qwen_loopback.py}"

process_is_local_qwen() {
  local candidate_pid="$1"
  local command_line
  [[ -r "/proc/$candidate_pid/cmdline" ]] || return 1
  [[ "$(readlink -f "/proc/$candidate_pid/cwd" 2>/dev/null)" == "$PROJECT_ROOT" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/$candidate_pid/cmdline")"
  [[ "$command_line" == *"$(basename "$SERVE_ENTRYPOINT")"* ]]
}

if [[ ! -s "$PID_FILE" ]]; then
  echo "status=stopped pid_file=$PID_FILE"
  exit 1
fi
server_pid="$(tr -dc '0-9' < "$PID_FILE")"
if [[ -z "$server_pid" ]] || ! kill -0 "$server_pid" 2>/dev/null; then
  echo "status=stale pid=${server_pid:-unknown}"
  exit 1
fi
if ! process_is_local_qwen "$server_pid"; then
  echo "status=unsafe_pid_file pid=$server_pid"
  exit 3
fi
if ! curl --fail --silent --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "status=starting_or_unhealthy pid=$server_pid endpoint=http://127.0.0.1:${PORT}/v1"
  exit 2
fi
echo "status=ready pid=$server_pid endpoint=http://127.0.0.1:${PORT}/v1"
