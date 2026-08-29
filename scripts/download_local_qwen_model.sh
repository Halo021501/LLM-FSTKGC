#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd -P)"
if [[ -f .env.qwen_local ]]; then
  set -a
  source .env.qwen_local
  set +a
fi
QWEN_PYTHON="${QWEN_PYTHON:-python}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
MODEL_REVISION="${MODEL_REVISION:-b25037543e9394b818fdfca67ab2a00ecc7dd641}"
MODEL_DIR="${LOCAL_QWEN_MODEL_DIR:-$PROJECT_ROOT/models/Qwen2.5-7B-Instruct-AWQ}"

if [[ ! -x "$QWEN_PYTHON" ]]; then
  echo "Missing qwen_local Python: $QWEN_PYTHON" >&2
  exit 2
fi

mkdir -p "$MODEL_DIR"
"$QWEN_PYTHON" -m huggingface_hub.commands.huggingface_cli download \
  "$MODEL_REPO" \
  --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR"

required=(config.json tokenizer_config.json tokenizer.json)
for filename in "${required[@]}"; do
  if [[ ! -s "$MODEL_DIR/$filename" ]]; then
    echo "Model download incomplete: missing $MODEL_DIR/$filename" >&2
    exit 3
  fi
done
shopt -s nullglob
weight_files=("$MODEL_DIR"/*.safetensors)
if (( ${#weight_files[@]} == 0 )); then
  echo "Model download incomplete: no safetensors weights in $MODEL_DIR" >&2
  exit 3
fi
if ! grep -Eq '"quant_method"[[:space:]]*:[[:space:]]*"awq"' "$MODEL_DIR/config.json"; then
  echo "Unexpected model config: AWQ quantization metadata is missing." >&2
  exit 3
fi

printf 'repository=%s\nrevision=%s\n' \
  "$MODEL_REPO" "$MODEL_REVISION" > "$MODEL_DIR/MODEL_SOURCE.txt"

(
  cd "$MODEL_DIR"
  find . -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.safetensors' -o -name '*.model' -o -name '*.txt' \) \
    -print0 | sort -z | xargs -0 sha256sum > MODEL_MANIFEST.sha256
)
echo "Local model ready: $MODEL_DIR"
echo "Repository: $MODEL_REPO revision=$MODEL_REVISION"
