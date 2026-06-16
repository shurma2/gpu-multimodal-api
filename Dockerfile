# syntax=docker/dockerfile:1.6
#
# Single-container image serving two models behind one OpenAI-compatible API:
#   - Gemma 4 12B (QAT Q4_0 GGUF) via llama.cpp  -> chat, vision, audio-in
#   - Qwen3-TTS-12Hz-0.6B-CustomVoice via PyTorch -> streaming TTS
#
# Stage 1 builds llama-server with CUDA; stage 2 is a slim CUDA runtime with the
# Python services. Target GPUs: Turing/Ampere/Ada (RTX 3060, A4000, 4060,
# 4070 Ti, ...).

ARG CUDA_VERSION=12.4.1
ARG UBUNTU=ubuntu22.04

# --------------------------------------------------------------------------- #
# Stage 1 — build llama.cpp (llama-server) with CUDA
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS llama-builder

ARG LLAMA_CPP_REF=master
# 75=Turing 80/86=Ampere(A4000,3060) 89=Ada(4060,4070Ti)
ARG CUDA_ARCHS="75;80;86;89"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ca-certificates \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" \
        https://github.com/ggml-org/llama.cpp /src/llama.cpp \
    || git clone https://github.com/ggml-org/llama.cpp /src/llama.cpp

RUN cmake -S /src/llama.cpp -B /src/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build /src/build --target llama-server -j"$(nproc)" \
    && mkdir -p /opt/llama/bin \
    && cp /src/build/bin/llama-server /opt/llama/bin/ \
    && find /src/build -name '*.so*' -exec cp -av {} /opt/llama/bin/ \;

# --------------------------------------------------------------------------- #
# Stage 2 — runtime
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        libcurl4 libgomp1 ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# llama-server + its shared libs
COPY --from=llama-builder /opt/llama/bin /opt/llama/bin
ENV PATH="/opt/llama/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama/bin:${LD_LIBRARY_PATH}"

# Python deps (torch from the CUDA 12.4 wheel index, then the rest)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
    && pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && pip3 install --no-cache-dir supervisor

WORKDIR /app
COPY gateway /app/gateway
COPY tts /app/tts
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
