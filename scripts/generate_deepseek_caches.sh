#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/ICEWS14}"
CACHE_DIR="${CACHE_DIR:-cache/standard_rolling_history}"

if [[ -f .env.deepseek ]]; then
  set -a
  source .env.deepseek
  set +a
fi

if [[ "${CONFIRM_DEEPSEEK_API_CALLS:-NO}" != "YES" ]]; then
  echo "Refusing API calls: set CONFIRM_DEEPSEEK_API_CALLS=YES in .env.deepseek after reviewing cost and data policy." >&2
  exit 2
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is empty. Copy .env.deepseek.example to .env.deepseek and fill it locally." >&2
  exit 2
fi

mkdir -p "$CACHE_DIR"
splits=(test)
if [[ "${INCLUDE_VALID:-NO}" == "YES" ]]; then
  splits=(valid test)
fi
echo "DeepSeek generation plan: shots=5,10 splits=${splits[*]}. Review API cost before continuing." >&2
for shot in 5 10; do
  for split in "${splits[@]}"; do
    "$PYTHON_BIN" scripts/stlp_generate_candidates.py \
      --data-dir "$DATA_DIR" \
      --split "$split" \
      --shot "$shot" \
      --seed 42 \
      --history-protocol standard_rolling_history \
      --provider deepseek \
      --execute-api \
      --resume \
      --output "$CACHE_DIR/${split}_s${shot}.jsonl"
  done
done
