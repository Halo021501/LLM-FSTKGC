#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/ICEWS14}"
ACTION="${1:-help}"
SCOPE="${REALTIME_SCOPE:-smoke}"
SPLIT="${REALTIME_SPLIT:-test}"
MODEL="${ALIYUN_QWEN_REALTIME_MODEL:-qwen-flash}"
ENV_FILE="${ALIYUN_QWEN_REALTIME_ENV_FILE:-.env.aliyun_qwen_realtime}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

# Re-read settings after the protected env file is loaded. Larger reviewed
# scopes receive more workers while remaining far below provider RPM/TPM caps.
MODEL="${ALIYUN_QWEN_REALTIME_MODEL:-qwen-flash}"
SPLIT="${REALTIME_SPLIT:-$SPLIT}"
if [[ ! "$SPLIT" =~ ^(test|valid)$ ]]; then
  echo "REALTIME_SPLIT must be test or valid." >&2
  exit 2
fi
export REALTIME_SPLIT="$SPLIT"
case "$SCOPE" in
  smoke)
    DEFAULT_WORKERS=8
    DEFAULT_MAX_RPM=240
    DEFAULT_MAX_TPM=300000
    ;;
  pilot)
    DEFAULT_WORKERS=16
    DEFAULT_MAX_RPM=480
    DEFAULT_MAX_TPM=600000
    ;;
  full)
    DEFAULT_WORKERS=32
    DEFAULT_MAX_RPM=960
    DEFAULT_MAX_TPM=1200000
    ;;
  *)
    DEFAULT_WORKERS=8
    DEFAULT_MAX_RPM=240
    DEFAULT_MAX_TPM=300000
    ;;
esac
TOTAL_WORKERS="${ALIYUN_QWEN_REALTIME_WORKERS:-$DEFAULT_WORKERS}"
TOTAL_MAX_RPM="${ALIYUN_QWEN_REALTIME_MAX_RPM:-$DEFAULT_MAX_RPM}"
TOTAL_MAX_TPM="${ALIYUN_QWEN_REALTIME_MAX_TPM:-$DEFAULT_MAX_TPM}"
TOKEN_RESERVATION="${ALIYUN_QWEN_REALTIME_TOKEN_RESERVATION:-1200}"
MAX_ATTEMPTS="${ALIYUN_QWEN_REALTIME_MAX_ATTEMPTS:-5}"
INSPECTION_MAX_ATTEMPTS="${ALIYUN_QWEN_REALTIME_INSPECTION_MAX_ATTEMPTS:-3}"
HEARTBEAT_SECONDS="${ALIYUN_QWEN_REALTIME_HEARTBEAT_SECONDS:-30}"
SOURCE_ROOT="${SOURCE_ROOT:-runs/aliyun_qwen_request_plans/${MODEL}/standard/${SCOPE}}"
RUN_ROOT="${RUN_ROOT:-runs/aliyun_qwen_realtime/${MODEL}/standard/${SCOPE}}"
CACHE_DIR="${CACHE_DIR:-cache/standard_rolling_history/aliyun_qwen_realtime/${MODEL}/standard}"

read -r -a shots <<<"${REALTIME_SHOTS:-5 10}"
if (( ${#shots[@]} == 0 )); then
  echo "REALTIME_SHOTS must contain at least one shot." >&2
  exit 2
fi
previous_shot=0
for shot in "${shots[@]}"; do
  if [[ ! "$shot" =~ ^(1|3|5|10)$ ]]; then
    echo "REALTIME_SHOTS permits only the ordered values 1, 3, 5, and 10." >&2
    exit 2
  fi
  if (( shot <= previous_shot )); then
    echo "REALTIME_SHOTS must be unique and strictly increasing." >&2
    exit 2
  fi
  previous_shot="$shot"
done
REALTIME_SHOTS="${shots[*]}"
export REALTIME_SHOTS
SHOT_COUNT="${#shots[@]}"
SHOT_TAG="$(IFS=_; printf '%s' "${shots[*]}")"
if [[ "$SPLIT" != "test" ]]; then
  SHOT_TAG="${SPLIT}_${SHOT_TAG}"
fi
if [[ "$SHOT_TAG" == "5_10" ]]; then
  CONTROLLER_LOG="${RUN_ROOT}/controller.log"
  CONTROLLER_PID="${RUN_ROOT}/controller.pid"
else
  CONTROLLER_LOG="${RUN_ROOT}/controller_${SHOT_TAG}.log"
  CONTROLLER_PID="${RUN_ROOT}/controller_${SHOT_TAG}.pid"
fi

if [[ "$MODEL" != "qwen-flash" ]]; then
  echo "The reviewed realtime path currently permits only qwen-flash." >&2
  exit 2
fi
if [[ ! "$SCOPE" =~ ^(smoke|pilot|full)$ ]]; then
  echo "REALTIME_SCOPE must be smoke, pilot, or full." >&2
  exit 2
fi
for value in "$TOTAL_WORKERS" "$TOTAL_MAX_RPM" "$TOTAL_MAX_TPM" "$TOKEN_RESERVATION" "$MAX_ATTEMPTS" "$INSPECTION_MAX_ATTEMPTS" "$HEARTBEAT_SECONDS"; do
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "Realtime concurrency/rate settings must be positive integers." >&2
    exit 2
  fi
done
if (( TOTAL_WORKERS > 64 || TOTAL_MAX_RPM > 30000 || TOTAL_MAX_TPM > 10000000 )); then
  echo "Realtime settings exceed the reviewed/documented safety ceiling." >&2
  exit 2
fi
if (( INSPECTION_MAX_ATTEMPTS > MAX_ATTEMPTS )); then
  echo "Inspection retries cannot exceed the general retry limit." >&2
  exit 2
fi

PER_WORKERS=$(( TOTAL_WORKERS / SHOT_COUNT ))
PER_RPM=$(( TOTAL_MAX_RPM / SHOT_COUNT ))
PER_TPM=$(( TOTAL_MAX_TPM / SHOT_COUNT ))
(( PER_WORKERS >= 1 )) || PER_WORKERS=1
(( PER_RPM >= 1 )) || PER_RPM=1
(( PER_TPM >= TOKEN_RESERVATION )) || {
  echo "Per-shot TPM budget is smaller than one token reservation." >&2
  exit 2
}

require_realtime_confirmation() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing protected credential file; expected $ENV_FILE." >&2
    exit 2
  fi
  local env_mode
  env_mode="$(stat -c '%a' "$ENV_FILE")"
  if (( (8#$env_mode & 077) != 0 )); then
    echo "Refusing credential use from $ENV_FILE with mode $env_mode; run chmod 600." >&2
    exit 2
  fi
  if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
    echo "DASHSCOPE_API_KEY is empty in the protected environment file." >&2
    exit 2
  fi
  if [[ "${CONFIRM_ALIYUN_QWEN_DATA_UPLOAD:-NO}" != "YES" ]] \
    || [[ "${CONFIRM_ALIYUN_QWEN_PAID_REALTIME:-NO}" != "YES" ]]; then
    echo "Realtime execution requires both upload and paid-realtime confirmations." >&2
    exit 2
  fi
}

source_job() {
  printf '%s/%s_s%s' "$SOURCE_ROOT" "$SPLIT" "$1"
}

run_dir() {
  printf '%s/%s_s%s' "$RUN_ROOT" "$SPLIT" "$1"
}

prepare_all() {
  unset DASHSCOPE_API_KEY
  local limit=0
  case "$SCOPE" in
    smoke) limit="${REALTIME_PLAN_LIMIT:-25}" ;;
    pilot) limit="${REALTIME_PLAN_LIMIT:-250}" ;;
    full) limit="${REALTIME_PLAN_LIMIT:-0}" ;;
  esac
  local shot
  for shot in "${shots[@]}"; do
    local -a args=(
      "$PYTHON_BIN" scripts/stlp_aliyun_qwen_request_plan.py
      --job-dir "$(source_job "$shot")"
      --data-dir "$DATA_DIR"
      --split "$SPLIT"
      --shot "$shot"
      --model "$MODEL"
      --limit "$limit"
    )
    [[ "${OMIT_SUPPORT:-NO}" == "YES" ]] && args+=(--omit-support)
    [[ "${OMIT_HISTORY:-NO}" == "YES" ]] && args+=(--omit-history)
    [[ "${PERMUTE_SUPPORT_ORDER:-NO}" == "YES" ]] && args+=(--permute-support-order)
    [[ "${REPLACE_ENTITY_NAMES:-NO}" == "YES" ]] && args+=(--replace-entity-names)
    "${args[@]}"
  done
}

run_one() {
  local shot="$1"
  local output_dir
  output_dir="$(run_dir "$shot")"
  mkdir -p "$output_dir"
  chmod 700 "$output_dir"
  umask 077
  "$PYTHON_BIN" scripts/stlp_aliyun_qwen_realtime.py run \
    --source-job-dir "$(source_job "$shot")" \
    --run-dir "$output_dir" \
    --model "$MODEL" \
    --workers "$PER_WORKERS" \
    --max-rpm "$PER_RPM" \
    --max-tpm "$PER_TPM" \
    --token-reservation "$TOKEN_RESERVATION" \
    --max-attempts "$MAX_ATTEMPTS" \
    --inspection-max-attempts "$INSPECTION_MAX_ATTEMPTS" \
    --heartbeat-seconds "$HEARTBEAT_SECONDS" \
    --resume --execute-api
}

collect_one() {
  local shot="$1"
  local output
  if [[ "$SCOPE" == "full" ]]; then
    output="$CACHE_DIR/${SPLIT}_s${shot}.jsonl"
  else
    output="$RUN_ROOT/collected/${SPLIT}_s${shot}.jsonl"
  fi
  mkdir -p "$(dirname "$output")"
  local -a args=(
    "$PYTHON_BIN" scripts/stlp_aliyun_qwen_realtime.py collect
    --source-job-dir "$(source_job "$shot")"
    --run-dir "$(run_dir "$shot")"
    --data-dir "$DATA_DIR"
    --output "$output"
  )
  if [[ "$SCOPE" != "full" ]]; then
    args+=(--allow-incomplete-cache)
  fi
  env -u DASHSCOPE_API_KEY "${args[@]}"
}

run_all() {
  require_realtime_confirmation
  mkdir -p "$RUN_ROOT"
  chmod 700 "$RUN_ROOT"
  umask 077
  local -a pids=()
  local shot
  for shot in "${shots[@]}"; do
    mkdir -p "$(run_dir "$shot")"
    chmod 700 "$(run_dir "$shot")"
    run_one "$shot" >"$(run_dir "$shot")/run.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "At least one selected realtime shot failed; inspect the sanitized run logs and resume." >&2
    return 1
  fi
  for shot in "${shots[@]}"; do
    collect_one "$shot"
  done
}

case "$ACTION" in
  prepare)
    prepare_all
    ;;
  run)
    run_all
    ;;
  start)
    require_realtime_confirmation
    mkdir -p "$RUN_ROOT"
    chmod 700 "$RUN_ROOT"
    umask 077
    if [[ -s "$CONTROLLER_PID" ]]; then
      existing_pid="$(<"$CONTROLLER_PID")"
      if [[ "$existing_pid" =~ ^[0-9]+$ ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "Realtime controller is already running with pid=$existing_pid." >&2
        exit 2
      fi
    fi
    nohup setsid --wait "$0" run </dev/null >>"$CONTROLLER_LOG" 2>&1 &
    controller_pid="$!"
    printf '%s\n' "$controller_pid" >"$CONTROLLER_PID"
    chmod 600 "$CONTROLLER_PID" "$CONTROLLER_LOG" 2>/dev/null || true
    echo "Started realtime controller pid=$controller_pid log=$CONTROLLER_LOG"
    ;;
  collect)
    unset DASHSCOPE_API_KEY
    for shot in "${shots[@]}"; do
      collect_one "$shot"
    done
    ;;
  estimate)
    unset DASHSCOPE_API_KEY
    for shot in "${shots[@]}"; do
      "$PYTHON_BIN" scripts/stlp_aliyun_qwen_realtime.py estimate \
        --source-job-dir "$(source_job "$shot")"
    done
    ;;
  status)
    unset DASHSCOPE_API_KEY
    for shot in "${shots[@]}"; do
      echo "shot=$shot"
      if [[ -f "$(run_dir "$shot")/realtime_state.json" ]]; then
        "$PYTHON_BIN" scripts/stlp_aliyun_qwen_realtime.py status \
          --run-dir "$(run_dir "$shot")"
      else
        echo '{"status":"not_started"}'
      fi
    done
    ;;
  help|-h|--help)
    cat <<'EOF'
	Usage: ./scripts/generate_aliyun_qwen_realtime_caches.sh ACTION

	  prepare  Build immutable target-blind request plans offline.
	  start    Launch the selected shots under a detached durable controller.
  run      Run/resume selected shots concurrently, then collect validated caches.
  status   Read local progress only; no key and no network access.
  collect  Rebuild caches offline after completed raw responses.
  estimate Estimate realtime list-price cost offline from the reviewed plan.

Scope defaults are smoke=8 workers/240 RPM/300K TPM, pilot=16/480/600K,
and full=32/960/1.2M, with 1200 reserved tokens/request, 5 general attempts,
at most 3 attempts before recording a provider content-inspection abstention,
and a 30-second main-consumer heartbeat. The in-flight window is bounded by the
per-shot worker count.
Budgets are divided equally between the selected shots. Set
REALTIME_SHOTS="1 3" to run the added low-shot conditions under an independent
controller_1_3.pid/log while the default REALTIME_SHOTS="5 10" controller runs.
	Set REALTIME_SCOPE=smoke|pilot|full. Run prepare before estimate/start; the
	resulting request/index artifacts are offline inputs to the realtime provider.
	Set REALTIME_SPLIT=test|valid to choose the evaluated split. The valid split
	reads the reviewed valid_s* request plans, writes valid_s* run directories and
caches, and uses controller_valid_<shots>.pid/log, so it never shares state or
output files with the test split.
EOF
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
