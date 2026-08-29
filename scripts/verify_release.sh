#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN_PYTHON="${QWEN_PYTHON:-python}"

sha256sum -c SOURCE_MANIFEST.sha256
bash -n \
  scripts/generate_deepseek_caches.sh \
  scripts/download_local_qwen_model.sh \
  scripts/start_local_qwen_server.sh \
  scripts/stop_local_qwen_server.sh \
  scripts/check_local_qwen_server.sh \
  scripts/generate_local_qwen_caches.sh \
  scripts/start_dynamic_qwen_full_generation.sh \
  scripts/guard_shared_qwen_worker.sh \
  scripts/generate_aliyun_qwen_batch_caches.sh \
  scripts/generate_aliyun_qwen_realtime_caches.sh \
  scripts/check_dynamic_qwen_pool.sh \
  scripts/configure_gpu_exclusive_protection.sh \
  scripts/restore_gpu_default_mode.sh \
  scripts/run_aliyun_qwen_realtime_frozen_task.sh \
  scripts/launch_aliyun_qwen_realtime_frozen_v5_matrix.sh \
  scripts/check_aliyun_qwen_realtime_frozen_v5_matrix.sh \
  scripts/run_llm_main_matrix.sh \
  scripts/evaluate_v5_llm_checkpoint_matrix.sh
if "$QWEN_PYTHON" -c 'import pynvml, uvloop, vllm' >/dev/null 2>&1; then
  "$QWEN_PYTHON" scripts/serve_local_qwen_loopback.py --help >/dev/null
else
  echo "Optional local-Qwen CLI check skipped; set QWEN_PYTHON to the pinned qwen_local interpreter."
fi
"$PYTHON_BIN" train.py --help >/dev/null
"$PYTHON_BIN" scripts/stlp_generate_candidates.py --help >/dev/null
"$PYTHON_BIN" scripts/stlp_aliyun_qwen_batch.py --help >/dev/null
"$PYTHON_BIN" scripts/stlp_aliyun_qwen_realtime.py --help >/dev/null
"$PYTHON_BIN" scripts/stlp_evaluate_llm_only.py --help >/dev/null
"$PYTHON_BIN" scripts/collect_aliyun_qwen_realtime_frozen_v5.py --help >/dev/null
"$PYTHON_BIN" scripts/dynamic_local_qwen_pool.py --help >/dev/null
"$PYTHON_BIN" scripts/attach_shared_qwen_worker.py --help >/dev/null
"$PYTHON_BIN" scripts/requeue_dynamic_qwen_failed_tasks.py --help >/dev/null
"$PYTHON_BIN" scripts/migrate_dynamic_qwen_worker_count.py --help >/dev/null
"$PYTHON_BIN" scripts/opportunistic_shared_qwen_supervisor.py --help >/dev/null
"$PYTHON_BIN" scripts/guard_qwen_scale4_controller.py --help >/dev/null
"$PYTHON_BIN" -m json.tool deploy/dynamic_qwen_gpu_services.json >/dev/null
"$PYTHON_BIN" -m json.tool ALIYUN_QWEN_BATCH_PROVENANCE.json >/dev/null
"$PYTHON_BIN" -m json.tool ALIYUN_QWEN_REALTIME_PROVENANCE.json >/dev/null
"$PYTHON_BIN" -m json.tool LLM_EXTENSION_PROVENANCE.json >/dev/null
"$PYTHON_BIN" -m compileall -q src scripts tests
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_aliyun_qwen*.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_local_qwen_abstention.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_qwen_scale4_controller_guard.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_shared_qwen_supervisor.py' -v
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_v170alterego_v5_llm_invariants.py' -v
