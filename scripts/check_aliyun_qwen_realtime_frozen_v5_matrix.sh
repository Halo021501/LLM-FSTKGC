#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ROOT="${1:-$ROOT/runs/aliyun_qwen_realtime_qwen_flash_20260809}"

echo "run_root=$RUN_ROOT"
if [[ -f "$RUN_ROOT/PIPELINE_DONE" ]]; then
  echo "pipeline=completed"
elif [[ -f "$RUN_ROOT/PIPELINE_FAILED" ]]; then
  echo "pipeline=failed"
  cat "$RUN_ROOT/PIPELINE_FAILED"
else
  echo "pipeline=running_or_waiting"
fi
find "$RUN_ROOT/frozen" -mindepth 2 -maxdepth 2 -name exit_code.txt -type f 2>/dev/null \
  | while read -r path; do
      [[ "$(tr -dc '0-9' < "$path")" == "0" ]] && printf '%s\n' "$path"
    done \
  | wc -l \
  | awk '{print "completed_tasks=" $1}'

if [[ -f "$RUN_ROOT/matrix.status" ]]; then
  tail -n 20 "$RUN_ROOT/matrix.status"
fi
