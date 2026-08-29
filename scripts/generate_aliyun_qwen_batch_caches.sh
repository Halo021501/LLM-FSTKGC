#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/ICEWS14}"
ENV_FILE="${ALIYUN_QWEN_BATCH_ENV_FILE:-.env.aliyun_qwen_batch}"
ACTION="${1:-help}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

MODEL="${ALIYUN_QWEN_BATCH_MODEL:-qwen3.7-flash-2026-07-15}"
if [[ ! "$MODEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Refusing unsafe model name for artifact paths: $MODEL" >&2
  exit 2
fi

case "$ACTION" in
  prepare-smoke)
    SCOPE="${BATCH_SCOPE:-smoke}"
    LIMIT="${SMOKE_LIMIT:-25}"
    ;;
  prepare-pilot)
    SCOPE="${BATCH_SCOPE:-pilot}"
    LIMIT="${PILOT_LIMIT:-250}"
    ;;
  prepare-full)
    SCOPE="${BATCH_SCOPE:-full}"
    LIMIT=0
    ;;
  *)
    SCOPE="${BATCH_SCOPE:-pilot}"
    LIMIT=0
    ;;
esac

# Offline children do not receive the credential even when the local env file
# contains one. This keeps request preparation, estimation, collection, and
# retry construction independently testable with network access disabled.
case "$ACTION" in
  submit|status|download|cancel|cleanup-remote) ;;
  *) unset DASHSCOPE_API_KEY ;;
esac

VARIANT="standard"
PROMPT_FLAGS=()
if [[ "${OMIT_SUPPORT:-NO}" == "YES" ]]; then
  VARIANT="${VARIANT}_no_support"
  PROMPT_FLAGS+=(--omit-support)
fi
if [[ "${OMIT_HISTORY:-NO}" == "YES" ]]; then
  VARIANT="${VARIANT}_no_history"
  PROMPT_FLAGS+=(--omit-history)
fi

JOB_ROOT="${JOB_ROOT:-runs/aliyun_qwen_batch/${MODEL}/${VARIANT}/${SCOPE}}"
CACHE_DIR="${CACHE_DIR:-cache/standard_rolling_history/aliyun_qwen_batch/${MODEL}/${VARIANT}}"

splits=(test)
if [[ "${INCLUDE_VALID:-NO}" == "YES" ]]; then
  splits=(valid test)
fi
read -r -a shots <<<"${BATCH_SHOTS:-5 10}"
if (( ${#shots[@]} == 0 )); then
  echo "BATCH_SHOTS must contain at least one shot." >&2
  exit 2
fi
previous_shot=0
for shot in "${shots[@]}"; do
  if [[ ! "$shot" =~ ^(1|3|5|10)$ ]]; then
    echo "BATCH_SHOTS permits only the ordered values 1, 3, 5, and 10." >&2
    exit 2
  fi
  if (( shot <= previous_shot )); then
    echo "BATCH_SHOTS must be unique and strictly increasing." >&2
    exit 2
  fi
  previous_shot="$shot"
done

job_dir_for() {
  local split="$1"
  local shot="$2"
  printf '%s/%s_s%s' "$JOB_ROOT" "$split" "$shot"
}

require_external_confirmation() {
  if [[ "${CONFIRM_ALIYUN_QWEN_DATA_UPLOAD:-NO}" != "YES" ]] \
    || [[ "${CONFIRM_ALIYUN_QWEN_PAID_BATCH:-NO}" != "YES" ]]; then
    echo "Refusing external API operation. Set both" >&2
    echo "  CONFIRM_ALIYUN_QWEN_DATA_UPLOAD=YES" >&2
    echo "  CONFIRM_ALIYUN_QWEN_PAID_BATCH=YES" >&2
    echo "only after reviewing the prepared request count, cost, and data policy." >&2
    exit 2
  fi
  if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "DASHSCOPE_API_KEY is empty; fill it only in the gitignored local env file." >&2
    exit 2
  fi
  if [[ -f "$ENV_FILE" ]]; then
    local env_mode
    env_mode="$(stat -c '%a' "$ENV_FILE")"
    if (( (8#$env_mode & 077) != 0 )); then
      echo "Refusing credential use from $ENV_FILE with mode $env_mode; run chmod 600 '$ENV_FILE'." >&2
      exit 2
    fi
  fi
}

for_each_job() {
  local operation="$1"
  local split shot job_dir output retry_dir
  local -a prepare_cmd
  for shot in "${shots[@]}"; do
    for split in "${splits[@]}"; do
      job_dir="$(job_dir_for "$split" "$shot")"
      case "$operation" in
        prepare)
          mkdir -p "$job_dir"
          prepare_cmd=(
            "$PYTHON_BIN" scripts/stlp_aliyun_qwen_batch.py prepare
            --job-dir "$job_dir"
            --data-dir "$DATA_DIR"
            --split "$split"
            --shot "$shot"
            --model "$MODEL"
          )
          if (( LIMIT > 0 )); then
            prepare_cmd+=(--limit "$LIMIT")
          fi
          prepare_cmd+=("${PROMPT_FLAGS[@]}")
          "${prepare_cmd[@]}"
          ;;
        estimate)
          "$PYTHON_BIN" scripts/stlp_aliyun_qwen_batch.py estimate \
            --job-dir "$job_dir"
          ;;
        submit|status|download|cancel)
          if [[ "$operation" == "submit" ]]; then
            "$PYTHON_BIN" scripts/stlp_aliyun_qwen_batch.py "$operation" \
              --job-dir "$job_dir" --execute-api \
              --completion-window "${ALIYUN_QWEN_BATCH_COMPLETION_WINDOW:-24h}"
          else
            "$PYTHON_BIN" scripts/stlp_aliyun_qwen_batch.py "$operation" \
              --job-dir "$job_dir" --execute-api
          fi
          ;;
        collect)
          if [[ "$SCOPE" == "full" ]]; then
            output="$CACHE_DIR/${split}_s${shot}.jsonl"
          else
            output="$JOB_ROOT/collected/${split}_s${shot}.jsonl"
          fi
          mkdir -p "$(dirname "$output")"
          collect_args=(
            "$PYTHON_BIN" scripts/stlp_aliyun_qwen_batch.py collect
            --job-dir "$job_dir" --data-dir "$DATA_DIR" --output "$output"
          )
          if [[ "$SCOPE" != "full" ]]; then
            collect_args+=(--allow-incomplete-cache)
          fi
          "${collect_args[@]}"
          ;;
        prepare-retry)
          retry_dir="$job_dir/retries/$(date -u +%Y%m%dT%H%M%SZ)"
          "$PYTHON_BIN" scripts/stlp_aliyun_qwen_batch.py prepare-retry \
            --job-dir "$job_dir" --retry-dir "$retry_dir"
          ;;
        cleanup-remote)
          "$PYTHON_BIN" scripts/stlp_aliyun_qwen_batch.py cleanup-remote \
            --job-dir "$job_dir" --execute-api --confirm-delete-remote
          ;;
        *)
          echo "internal error: unknown operation $operation" >&2
          exit 2
          ;;
      esac
    done
  done
}

case "$ACTION" in
  prepare-smoke|prepare-pilot|prepare-full)
    echo "Offline prepare: model=$MODEL scope=$SCOPE variant=$VARIANT limit=$LIMIT" >&2
    echo "No API key is read by the Python prepare operation and no network call is authorized." >&2
    for_each_job prepare
    echo "Prepared requests under $JOB_ROOT." >&2
    echo "Run 'BATCH_SCOPE=$SCOPE $0 estimate' and inspect every plan before submission." >&2
    ;;
  estimate)
    for_each_job estimate
    ;;
  submit|status|download|cancel)
    require_external_confirmation
    echo "Authorized external operation: action=$ACTION model=$MODEL scope=$SCOPE jobs=$JOB_ROOT" >&2
    for_each_job "$ACTION"
    ;;
  collect)
    echo "Offline collection: scope=$SCOPE jobs=$JOB_ROOT" >&2
    for_each_job collect
    ;;
  prepare-retry)
    echo "Offline retry preparation: scope=$SCOPE jobs=$JOB_ROOT" >&2
    for_each_job prepare-retry
    ;;
  cleanup-remote)
    require_external_confirmation
    if [[ "${CONFIRM_ALIYUN_QWEN_REMOTE_DELETE:-NO}" != "YES" ]]; then
      echo "Refusing remote deletion: set CONFIRM_ALIYUN_QWEN_REMOTE_DELETE=YES only after local hash verification." >&2
      exit 2
    fi
    for_each_job cleanup-remote
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: ./scripts/generate_aliyun_qwen_batch_caches.sh ACTION

Offline actions (never authorize API access):
  prepare-smoke    Prepare 25 queries per split/shot (50 test requests total).
  prepare-pilot    Prepare 250 queries per split/shot (500 test requests total).
  prepare-full     Prepare complete split/shot request files.
  estimate         Summarize prepared request counts and token/cost inputs.
  collect          Validate downloaded results and build provider-specific caches.
  prepare-retry    Prepare only failed, missing, or invalid requests for retry.

External actions (require both upload and paid-batch confirmations):
  submit           Upload request files and create Batch jobs.
  status           Query Batch job state.
  download         Download completed output/error files.
  cancel           Cancel queued/running Batch jobs and preserve local state.
  cleanup-remote   Delete remote files; also requires the deletion confirmation.

Important environment controls:
  BATCH_SCOPE=pilot|smoke|full   Select matching prepared job tree (default pilot).
  BATCH_SHOTS="1 3"              Select an ordered subset of 1, 3, 5, 10.
  PILOT_LIMIT=250 SMOKE_LIMIT=25
  INCLUDE_VALID=YES             Include validation caches after pilot approval.
  OMIT_SUPPORT=YES / OMIT_HISTORY=YES
  ALIYUN_QWEN_BATCH_MODEL=qwen3.7-flash-2026-07-15

The moving qwen-flash alias is accepted only as an explicit override and writes
to a distinct model directory. It is not the default paper condition.
EOF
    ;;
  *)
    echo "Unknown action: $ACTION. Run '$0 help'." >&2
    exit 2
    ;;
esac
