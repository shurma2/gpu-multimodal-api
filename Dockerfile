# syntax=docker/dockerfile:1.6
#
# Single-container image for GPU rental templates:
#   - llama.cpp server on 127.0.0.1:8081
#   - FastAPI voice gateway on 0.0.0.0:8000
#
# CUDA stays on 12.4 for RTX 3090 Ti hosts whose drivers are capped at 12.x.

ARG CUDA_VERSION=12.4.1
ARG UBUNTU=ubuntu22.04
ARG PYTORCH_IMAGE=pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# --------------------------------------------------------------------------- #
# Stage 1 - build llama.cpp with CUDA
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS llama-builder

ARG CUDA_ARCHS="86"
# Must be a llama.cpp release that supports the Gemma 4 architecture (>= b9724).
ARG LLAMA_CPP_REF=b9728
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ca-certificates \
        libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" \
        https://github.com/ggml-org/llama.cpp /src/llama.cpp

RUN cmake -S /src/llama.cpp -B /src/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DLLAMA_OPENSSL=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
    && cmake --build /src/build --target llama-server -j"$(nproc)" \
    && mkdir -p /opt/llama/bin /opt/llama/lib \
    && cp /src/build/bin/llama-server /opt/llama/bin/ \
    && find /src/build -name '*.so*' -exec cp -av {} /opt/llama/bin/ \;

# The runtime stage is a PyTorch image that keeps its CUDA libraries in a conda
# path llama-server cannot see, so the GPU build crashed with
# "libcudart.so.12: cannot open shared object file". Ship the exact CUDA 12.4
# runtime libraries llama-server linked against. libcuda.so (the driver) is NOT
# copied — it is injected from the host by the NVIDIA Container Toolkit.
RUN cp -av /usr/local/cuda/lib64/libcudart.so*   /opt/llama/lib/ \
    && cp -av /usr/local/cuda/lib64/libcublas.so*   /opt/llama/lib/ \
    && cp -av /usr/local/cuda/lib64/libcublasLt.so* /opt/llama/lib/

# --------------------------------------------------------------------------- #
# Stage 2 - runtime
# --------------------------------------------------------------------------- #
FROM ${PYTORCH_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        espeak-ng \
        ffmpeg \
        libcurl4 \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Bring over the binary, ggml shared libs (/opt/llama/bin) and the CUDA runtime
# libs (/opt/llama/lib). The CUDA libs are added to llama-server's library path
# in entrypoint.sh only, so they never shadow the PyTorch image's own CUDA libs.
COPY --from=llama-builder /opt/llama /opt/llama
ENV PATH="/opt/llama/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama/bin:${LD_LIBRARY_PATH}" \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        -r /tmp/requirements.txt \
        supervisor

WORKDIR /opt/app
COPY gateway /opt/app/gateway
COPY tts /opt/app/tts
COPY voice /opt/app/voice
COPY supervisord.conf /etc/supervisord.conf
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV HF_HOME=/models/hf-cache \
    HUGGINGFACE_HUB_CACHE=/models/hf-cache/hub \
    NEMO_CACHE_DIR=/models/nemo-cache \
    LLM_BASE_URL=http://127.0.0.1:8081 \
    LLM_MODEL_NAME=gemma-4-12b \
    LLM_HF_REPO=google/gemma-4-12B-it-qat-q4_0-gguf \
    LLM_HF_QUANT=Q4_0 \
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

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=10 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
