#!/usr/bin/env bash
# Container entrypoint.
#   no args  -> start supervisord (llama-server + tts + gateway)
#   "llama"  -> build the llama-server command from env and exec it
#               (invoked by supervisord; keeps the long flag list out of conf)
set -euo pipefail

if [[ "${1:-}" == "llama" ]]; then
    # Performance/VRAM knobs. On very new llama.cpp builds flash-attn takes a
    # value (e.g. "--flash-attn on"); override LLM_PERF_ARGS if you hit a parse
    # error. KV-cache quantisation (q8_0) roughly halves cache VRAM.
    PERF_ARGS="${LLM_PERF_ARGS:---flash-attn on --cache-type-k q8_0 --cache-type-v q8_0}"

    # Extra load-time flags. Empty by default (the pinned llama.cpp build is
    # known-good). On newer master that auto-"fits" params and crash-loops, set
    # LLM_FIT_ARGS="-fit off".
    FIT_ARGS="${LLM_FIT_ARGS:-}"

    # Locate the llama-server binary (the official base image ships it in /app).
    LLAMA_BIN="$(command -v llama-server || true)"
    if [[ -z "${LLAMA_BIN}" ]]; then
        for p in /app/llama-server /usr/local/bin/llama-server /llama-server; do
            [[ -x "$p" ]] && LLAMA_BIN="$p" && break
        done
    fi
    : "${LLAMA_BIN:?llama-server binary not found}"

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
