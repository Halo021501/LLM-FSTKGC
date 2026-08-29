#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [state_dir]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  STATE_DIR="$1"
elif [[ -s logs/dynamic_qwen_pool.latest ]]; then
  STATE_DIR="$(head -n 1 logs/dynamic_qwen_pool.latest)"
else
  echo "No state directory supplied and no latest-run pointer exists." >&2
  exit 2
fi

if [[ -s "$STATE_DIR/controller.pid" ]]; then
  controller_pid="$(tr -dc '0-9' < "$STATE_DIR/controller.pid")"
elif [[ -s "$STATE_DIR/launcher.pid" ]]; then
  controller_pid="$(tr -dc '0-9' < "$STATE_DIR/launcher.pid")"
else
  controller_pid=""
fi
if [[ -n "$controller_pid" ]] && kill -0 "$controller_pid" 2>/dev/null; then
  echo "controller=alive pid=$controller_pid"
else
  echo "controller=not_alive pid=${controller_pid:-unknown}"
fi

if [[ -s "$STATE_DIR/status.json" ]]; then
  "$PYTHON_BIN" -m json.tool "$STATE_DIR/status.json"
else
  echo "status=pending state_dir=$STATE_DIR"
  if [[ -s "$STATE_DIR/controller.log" ]]; then
    tail -n 40 "$STATE_DIR/controller.log"
  fi
fi
