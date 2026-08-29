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
  echo "No local Qwen PID file: $PID_FILE"
  exit 0
fi
server_pid="$(tr -dc '0-9' < "$PID_FILE")"
if [[ -z "$server_pid" ]] || ! kill -0 "$server_pid" 2>/dev/null; then
  echo "Local Qwen is not running; removing stale PID file."
  rm -f "$PID_FILE"
  exit 0
fi
if ! process_is_local_qwen "$server_pid"; then
  echo "PID file points to a live non-Qwen process; refusing to send a signal: $server_pid" >&2
  exit 3
fi

process_group="$(ps -o pgid= -p "$server_pid" | tr -dc '0-9')"
if [[ "$process_group" == "$server_pid" ]]; then
  kill -- "-$server_pid"
else
  kill "$server_pid"
fi
for _ in $(seq 1 30); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Local Qwen stopped."
    exit 0
  fi
  sleep 1
done
echo "Local Qwen did not stop within 30 seconds; manual inspection is required for PID $server_pid." >&2
exit 4
