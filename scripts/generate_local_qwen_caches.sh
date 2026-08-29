#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/ICEWS14}"
CACHE_DIR="${CACHE_DIR:-cache/standard_rolling_history/qwen2.5-7b-awq}"

if [[ -f .env.qwen_local ]]; then
  set -a
  source .env.qwen_local
  set +a
fi
PORT="${LOCAL_QWEN_PORT:-8000}"
export LOCAL_QWEN_BASE_URL="${LOCAL_QWEN_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
export LOCAL_QWEN_MODEL="${LOCAL_QWEN_MODEL:-Qwen2.5-7B-Instruct-AWQ}"

if [[ "${CONFIRM_LOCAL_QWEN_GENERATION:-NO}" != "YES" ]]; then
  echo "Refusing the 26,358-query run: set CONFIRM_LOCAL_QWEN_GENERATION=YES after reviewing the cache plan." >&2
  exit 2
fi
if ! ./scripts/check_local_qwen_server.sh >/dev/null; then
  echo "Local Qwen is not ready. Start it before cache generation." >&2
  exit 3
fi

mkdir -p "$CACHE_DIR"
splits=(test)
if [[ "${INCLUDE_VALID:-NO}" == "YES" ]]; then
  splits=(valid test)
fi
echo "Local Qwen generation plan: model=$LOCAL_QWEN_MODEL shots=5,10 splits=${splits[*]}" >&2
for shot in 5 10; do
  for split in "${splits[@]}"; do
    "$PYTHON_BIN" scripts/stlp_generate_candidates.py \
      --data-dir "$DATA_DIR" \
      --split "$split" \
      --shot "$shot" \
      --seed 42 \
      --history-protocol standard_rolling_history \
      --provider local_qwen \
      --max-tokens 512 \
      --timeout 180 \
      --max-retries 1 \
      --request-interval 0 \
      --resume \
      --output "$CACHE_DIR/${split}_s${shot}.jsonl"
  done
done
