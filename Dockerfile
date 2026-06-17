# syntax=docker/dockerfile:1.6
#
# Single-container image serving two models behind one OpenAI-compatible API:
#   - Gemma 4 12B (QAT Q4_0 GGUF) via llama.cpp  -> chat, vision, audio-in
#   - Qwen3-TTS-12Hz-0.6B-CustomVoice via PyTorch -> streaming TTS
#
# We start FROM the official prebuilt llama.cpp CUDA server image (multimodal /
# mtmd enabled, broad GPU-arch coverage, libcurl/-hf support) and layer the
# Python TTS service + gateway on top. No from-source llama.cpp compilation, so
# no CUDA driver-stub linking headaches.

ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda
FROM ${LLAMA_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# llama-server and its shared libs live in /app in the base image.
ENV PATH="/app:${PATH}" \
    LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"

# Python deps (torch from the CUDA 12.4 wheel index, then the rest).
# The base is Ubuntu 24.04 / Python 3.12 with PEP-668 "externally managed"
# protection; inside a container we install system-wide on purpose.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore
COPY requirements.txt /tmp/requirements.txt
# Note: do NOT `--upgrade pip` here — the distro pip has no RECORD file and the
# self-upgrade fails. The packaged pip installs wheels fine.
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir -r /tmp/requirements.txt supervisor

# Our services (kept out of /app so they don't mix with llama.cpp binaries).
WORKDIR /opt/app
COPY gateway /opt/app/gateway
COPY tts /opt/app/tts
COPY supervisord.conf /etc/supervisord.conf
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Weights + HF cache live on the mounted /models volume.
ENV HF_HOME=/models/hf-cache \
    LLM_BASE_URL=http://127.0.0.1:8081 \
    TTS_BASE_URL=http://127.0.0.1:8082 \
    LLM_MODEL_NAME=gemma-4-12b \
    TTS_MODEL_NAME=qwen3-tts \
    TTS_DEFAULT_SPEAKER=Ryan \
    TTS_DEFAULT_LANGUAGE=English

EXPOSE 8000
VOLUME ["/models"]

# start-period is generous: first boot downloads ~9 GB of weights.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=10 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
