# syntax=docker/dockerfile:1.6
#
# Single-container image serving two models behind one OpenAI-compatible API:
#   - Gemma 4 12B (QAT Q4_0 GGUF) via llama.cpp  -> chat, vision, audio-in
#   - Qwen3-TTS-12Hz-0.6B-CustomVoice via PyTorch -> streaming TTS
#
# Built against CUDA 12.4 on purpose: prebuilt "latest" images (llama.cpp,
# torchaudio) now target CUDA 13 and fail on rented hosts whose driver only
# supports CUDA 12.x ("unsupported display driver / cuda driver combination").
# 12.4 runs on any driver supporting CUDA >= 12.4.

ARG CUDA_VERSION=12.4.1
ARG UBUNTU=ubuntu22.04

# --------------------------------------------------------------------------- #
# Stage 1 — build llama.cpp (llama-server) with CUDA
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS llama-builder

# Ampere (A4000/A5000/3060=86, A100=80) + Ada (4060/4070Ti=89).
ARG CUDA_ARCHS="80;86;89"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ca-certificates \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# The devel image ships only the CUDA driver *stub* (libcuda.so); the real
# libcuda.so.1 is injected by the host at runtime. Make the stub linkable so
# the executable link step can resolve cuMem*/cuDevice* symbols.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /src/llama.cpp

RUN cmake -S /src/llama.cpp -B /src/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DLLAMA_CURL=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
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
        ffmpeg sox libsox-fmt-all \
        libcurl4 libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# llama-server + its shared libs.
COPY --from=llama-builder /opt/llama/bin /opt/llama/bin
ENV PATH="/opt/llama/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama/bin:${LD_LIBRARY_PATH}"

# Python deps.
#  - torch + torchaudio strictly from the CUDA 12.4 index (--index-url, not
#    extra) so we get a matching cu124 pair (mismatched torchaudio loads the
#    wrong libcudart and crashes).
#  - numpy + setuptools up front, then --no-build-isolation: the `sox` sdist
#    (pulled in by qwen-tts) imports numpy in its setup.py, which fails under
#    pip's default build isolation.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_ROOT_USER_ACTION=ignore
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchaudio \
    && pip3 install --no-cache-dir "numpy<2.0" setuptools wheel \
    && pip3 install --no-cache-dir --no-build-isolation -r /tmp/requirements.txt supervisor

# Our services (kept out of /opt/llama).
WORKDIR /opt/app
COPY gateway /opt/app/gateway
COPY tts /opt/app/tts
COPY supervisord.conf /etc/supervisord.conf
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV HF_HOME=/models/hf-cache \
    LLM_BASE_URL=http://127.0.0.1:8081 \
    TTS_BASE_URL=http://127.0.0.1:8082 \
    LLM_MODEL_NAME=gemma-4-12b \
    TTS_MODEL_NAME=qwen3-tts \
    TTS_DEFAULT_SPEAKER=Ryan \
    TTS_DEFAULT_LANGUAGE=English

EXPOSE 8000
VOLUME ["/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=10 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
