#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "llama" ]]; then
    PERF_ARGS="${LLM_PERF_ARGS:---flash-attn on --cache-type-k q8_0 --cache-type-v q8_0}"
    FIT_ARGS="${LLM_FIT_ARGS:-}"

    LLAMA_BIN="$(command -v llama-server || true)"
    : "${LLAMA_BIN:?llama-server binary not found}"

    exec "${LLAMA_BIN}" \
        -hf "${LLM_HF_REPO:-google/gemma-3-12b-it-qat-q4_0-gguf}:${LLM_HF_QUANT:-Q4_0}" \
        --host 127.0.0.1 --port 8081 \
        --alias "${LLM_MODEL_NAME:-gemma-3-12b}" \
        -ngl "${LLM_GPU_LAYERS:-99}" \
        -c "${LLM_CONTEXT:-8192}" \
        -np "${LLM_PARALLEL:-2}" \
        --jinja \
        ${FIT_ARGS} \
        ${PERF_ARGS} \
        ${LLM_EXTRA_ARGS:-}
fi

exec supervisord -c /etc/supervisord.conf
