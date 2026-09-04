#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
  echo "The selected PYTHON_BIN does not provide PyTorch; install requirements.txt or select the project environment." >&2
  exit 2
fi

sha256sum -c SOURCE_MANIFEST.sha256

bash -n \
  scripts/generate_aliyun_qwen_realtime_caches.sh \
  scripts/run_aliyun_qwen_realtime_frozen_task.sh \
  scripts/launch_aliyun_qwen_realtime_frozen_v5_matrix.sh \
  scripts/check_aliyun_qwen_realtime_frozen_v5_matrix.sh \
  scripts/evaluate_v5_llm_checkpoint_matrix.sh \
  scripts/run_toy_smoke.sh

"$PYTHON_BIN" train.py --help >/dev/null
"$PYTHON_BIN" scripts/stlp_aliyun_qwen_request_plan.py --help >/dev/null
"$PYTHON_BIN" scripts/stlp_aliyun_qwen_realtime.py --help >/dev/null
"$PYTHON_BIN" scripts/stlp_evaluate_llm_only.py --help >/dev/null
"$PYTHON_BIN" scripts/prepare_aliyun_qwen_realtime_shards.py --help >/dev/null
"$PYTHON_BIN" scripts/merge_aliyun_qwen_realtime_shards.py --help >/dev/null
"$PYTHON_BIN" scripts/collect_aliyun_qwen_realtime_frozen_v5.py --help >/dev/null
"$PYTHON_BIN" -m json.tool ALIYUN_QWEN_REALTIME_PROVENANCE.json >/dev/null

"$PYTHON_BIN" -m py_compile \
  src/*.py \
  scripts/stlp_aliyun_qwen_request_plan.py \
  scripts/stlp_aliyun_qwen_realtime.py \
  scripts/stlp_evaluate_llm_only.py \
  scripts/prepare_aliyun_qwen_realtime_shards.py \
  scripts/merge_aliyun_qwen_realtime_shards.py \
  scripts/collect_aliyun_qwen_realtime_frozen_v5.py \
  tests/test_aliyun_qwen_realtime.py \
  tests/test_aliyun_qwen_realtime_shards.py \
  tests/test_v170alterego_v5_llm_invariants.py

"$PYTHON_BIN" -m unittest discover -s tests -p 'test_aliyun_qwen_realtime*.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_v170alterego_v5_llm_invariants.py' -v
