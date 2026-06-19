#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "llama" ]]; then
    PERF_ARGS="${LLM_PERF_ARGS:---flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --no-mmproj}"
    FIT_ARGS="${LLM_FIT_ARGS:-}"

    LLAMA_BIN="$(command -v llama-server || true)"
    : "${LLAMA_BIN:?llama-server binary not found}"

    # Scope the CUDA 12.4 runtime libs shipped in /opt/llama/lib to llama-server
    # only, so they never shadow the PyTorch image's CUDA libs used by STT/TTS.
    export LD_LIBRARY_PATH="/opt/llama/lib:/opt/llama/bin:${LD_LIBRARY_PATH:-}"

    exec "${LLAMA_BIN}" \
        -hf "${LLM_HF_REPO:-google/gemma-4-12B-it-qat-q4_0-gguf}:${LLM_HF_QUANT:-Q4_0}" \
        --host 127.0.0.1 --port 8081 \
        --alias "${LLM_MODEL_NAME:-gemma-4-12b}" \
        -ngl "${LLM_GPU_LAYERS:-99}" \
        -c "${LLM_CONTEXT:-8192}" \
        -np "${LLM_PARALLEL:-2}" \
        --jinja \
        ${FIT_ARGS} \
        ${PERF_ARGS} \
        ${LLM_EXTRA_ARGS:-}
fi

exec supervisord -c /etc/supervisord.conf
