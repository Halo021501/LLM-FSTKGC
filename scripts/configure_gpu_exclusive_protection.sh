#!/usr/bin/env bash
set -euo pipefail

# NVIDIA compute mode is a host-wide access-control setting. This script must
# be run by an administrator and deliberately refuses to evict or coexist with
# an unknown CUDA process.

if [[ "${EUID}" -ne 0 ]]; then
  echo "Administrator privileges are required (run with sudo)." >&2
  exit 2
fi
if [[ "${CONFIRM_GPU_EXCLUSIVE_PROTECTION:-NO}" != "YES" ]]; then
  echo "Set CONFIRM_GPU_EXCLUSIVE_PROTECTION=YES to change host GPU compute mode." >&2
  exit 2
fi

TARGET_USER="${GPU_PROTECTION_USER:-${SUDO_USER:-${USER:-}}}"
GPU_LIST="${PROTECTED_GPU_IDS:-3,5}"
if [[ ! "$TARGET_USER" =~ ^[a-z_][a-z0-9_-]*[$]?$ ]]; then
  echo "Invalid GPU_PROTECTION_USER: $TARGET_USER" >&2
  exit 2
fi
if [[ ! "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "PROTECTED_GPU_IDS must be a comma-separated integer list." >&2
  exit 2
fi

IFS=',' read -r -a gpu_ids <<<"$GPU_LIST"
for gpu_id in "${gpu_ids[@]}"; do
  mapfile -t pids < <(
    /usr/bin/nvidia-smi -i "$gpu_id" \
      --query-compute-apps=pid --format=csv,noheader,nounits |
      /usr/bin/sed '/^[[:space:]]*$/d'
  )
  if [[ "${#pids[@]}" -ne 1 ]]; then
    echo "GPU $gpu_id has ${#pids[@]} CUDA processes; expected exactly one protected process." >&2
    exit 3
  fi
  pid="${pids[0]//[[:space:]]/}"
  owner="$(/usr/bin/ps -o user= -p "$pid" | /usr/bin/xargs)"
  if [[ "$owner" != "$TARGET_USER" ]]; then
    echo "GPU $gpu_id PID $pid belongs to $owner, not $TARGET_USER; refusing." >&2
    exit 3
  fi
  command_line="$(/usr/bin/ps -o args= -p "$pid")"
  if [[ "$command_line" != *"serve_local_qwen_loopback.py"* ]]; then
    echo "GPU $gpu_id PID $pid is not the expected local-Qwen service; refusing." >&2
    exit 3
  fi
done

changed_gpu_ids=()
rollback_changed_modes() {
  original_status="$?"
  set +e
  for changed_gpu_id in "${changed_gpu_ids[@]}"; do
    /usr/bin/nvidia-smi -i "$changed_gpu_id" -c DEFAULT >/dev/null
    echo "Rolled GPU $changed_gpu_id back to Default mode after a protection error." >&2
  done
  exit "$original_status"
}
trap rollback_changed_modes ERR

for gpu_id in "${gpu_ids[@]}"; do
  /usr/bin/nvidia-smi -i "$gpu_id" -c EXCLUSIVE_PROCESS
  changed_gpu_ids+=("$gpu_id")
  mode="$(
    /usr/bin/nvidia-smi -i "$gpu_id" --query-gpu=compute_mode \
      --format=csv,noheader,nounits | /usr/bin/xargs
  )"
  if [[ "$mode" != "Exclusive_Process" && "$mode" != "EXCLUSIVE_PROCESS" ]]; then
    echo "GPU $gpu_id did not enter Exclusive Process mode: $mode" >&2
    exit 4
  fi
  echo "GPU $gpu_id protected in Exclusive Process mode for existing $TARGET_USER Qwen PID."
done

trap - ERR
echo "Protection is host-wide and lasts until reset or reboot."
