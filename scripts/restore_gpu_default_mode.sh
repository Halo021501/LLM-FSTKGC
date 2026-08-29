#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Administrator privileges are required (run with sudo)." >&2
  exit 2
fi
if [[ "${CONFIRM_GPU_DEFAULT_RESTORE:-NO}" != "YES" ]]; then
  echo "Set CONFIRM_GPU_DEFAULT_RESTORE=YES to restore shared GPU compute mode." >&2
  exit 2
fi

GPU_LIST="${PROTECTED_GPU_IDS:-3,5}"
if [[ ! "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "PROTECTED_GPU_IDS must be a comma-separated integer list." >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<<"$GPU_LIST"
for gpu_id in "${gpu_ids[@]}"; do
  /usr/bin/nvidia-smi -i "$gpu_id" -c DEFAULT
  echo "GPU $gpu_id restored to Default compute mode."
done
