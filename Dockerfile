# syntax=docker/dockerfile:1.6
#
# Python backend image for the voice stack. The LLM runs in a separate
# ghcr.io/ggml-org/llama.cpp:server-cuda container from docker-compose.yml.
# CUDA is intentionally kept on the 12.4 line for rented RTX 3090 Ti hosts.

ARG PYTORCH_IMAGE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf-cache \
    HUGGINGFACE_HUB_CACHE=/models/hf-cache/hub \
    NEMO_CACHE_DIR=/models/nemo-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        espeak-ng \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r /tmp/requirements.txt

COPY gateway /opt/app/gateway
COPY tts /opt/app/tts
COPY voice /opt/app/voice

ENV LLM_BASE_URL=http://llm:8080 \
    LLM_MODEL_NAME=gemma-3-12b \
    TTS_MODEL_NAME=kokoro \
    TTS_DEFAULT_SPEAKER=am_michael \
    TTS_DEFAULT_LANGUAGE=English \
    STT_MODEL_NAME=nemotron-speech-streaming-en-0.6b \
    STT_MODEL_ID=nvidia/nemotron-speech-streaming-en-0.6b \
    STT_DEVICE=cuda \
    VAD_MODEL_NAME=silero-vad-onnx \
    TURN_MODEL_NAME=smart-turn-v3

EXPOSE 8000
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=10 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
