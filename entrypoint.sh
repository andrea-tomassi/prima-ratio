#!/usr/bin/env bash
# Download the Decision 3.0 mini GGUF package on first start, then hand over to
# llama.cpp's server (which exposes /v1/systemone for vllm-d3 decision models).
set -euo pipefail

MODEL_DIR="${PRIMA_CACHE}/models"
mkdir -p "$MODEL_DIR"

fetch() {
    local file="$1"
    local target="$MODEL_DIR/$file"
    if [ -s "$target" ]; then
        echo "[prima-ratio] ${file} already cached"
        return 0
    fi
    local url="https://huggingface.co/${PRIMA_MODEL_REPO}/resolve/${PRIMA_MODEL_REVISION}/${file}?download=true"
    echo "[prima-ratio] downloading ${file} (first start)…"
    curl -fL --retry 5 --retry-delay 3 --retry-all-errors -o "${target}.part" "$url"
    mv "${target}.part" "$target"
}

fetch "$PRIMA_MODEL_FILE"
fetch "$PRIMA_MMPROJ_FILE"

echo "[prima-ratio] starting llama-server (ctx=${PRIMA_CTX}, kv=${PRIMA_KV_TYPE}) on port ${PRIMA_PORT}"
# shellcheck disable=SC2086
exec /opt/llama/llama-server \
    -m "${MODEL_DIR}/${PRIMA_MODEL_FILE}" \
    --mmproj "${MODEL_DIR}/${PRIMA_MMPROJ_FILE}" \
    -c "${PRIMA_CTX}" -b "${PRIMA_CTX}" -ub "${PRIMA_CTX}" \
    -ctk "${PRIMA_KV_TYPE}" -ctv "${PRIMA_KV_TYPE}" \
    -ngl 99 -fa on \
    --host 0.0.0.0 --port "${PRIMA_PORT}" \
    ${PRIMA_EXTRA_ARGS:-}
