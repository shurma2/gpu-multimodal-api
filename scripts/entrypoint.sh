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
    PERF_ARGS="${LLM_PERF_ARGS:---flash-attn --cache-type-k q8_0 --cache-type-v q8_0}"

    exec llama-server \
        -hf "${LLM_HF_REPO:-google/gemma-4-12B-it-qat-q4_0-gguf}:${LLM_HF_QUANT:-Q4_0}" \
        --host 127.0.0.1 --port 8081 \
        --alias "${LLM_MODEL_NAME:-gemma-4-12b}" \
        -ngl "${LLM_GPU_LAYERS:-99}" \
        -c "${LLM_CONTEXT:-8192}" \
        -np "${LLM_PARALLEL:-2}" \
        --jinja \
        ${PERF_ARGS} \
        ${LLM_EXTRA_ARGS:-}
fi

exec supervisord -c /etc/supervisord.conf
