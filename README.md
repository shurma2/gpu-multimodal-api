# Speaker Voice API

Docker stack for a local voice agent backend on an NVIDIA GPU host, tuned for an RTX 3090 Ti / CUDA 12.x environment.

## Services

| Compose service | Image/base | Purpose |
| --- | --- | --- |
| `llm` | `ghcr.io/ggml-org/llama.cpp:server-cuda` | Gemma 3 12B Q4 GGUF, OpenAI-compatible llama.cpp API |
| `api` | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime` | FastAPI gateway, STT, VAD, Smart Turn, Kokoro TTS |

Only `api` publishes a port: `8000`.

## Models

| Role | Model | Runtime |
| --- | --- | --- |
| STT | `nvidia/nemotron-speech-streaming-en-0.6b` | NVIDIA NeMo, GPU |
| VAD | Silero VAD ONNX bundled with Pipecat | CPU |
| End of turn | Smart Turn v3.2 ONNX bundled with Pipecat | CPU |
| LLM | `google/gemma-3-12b-it-qat-q4_0-gguf:Q4_0` | llama.cpp server, GPU |
| TTS | Kokoro-82M, 24 kHz | GPU by default, CPU optional |

HF and NeMo caches are named Docker volumes, so first boot downloads once and later restarts reuse the weights.

## Requirements On The Host

- NVIDIA driver with CUDA 12.x support.
- NVIDIA Container Toolkit.
- Docker Compose v2.
- GPU with enough VRAM. RTX 3090 Ti 24 GB is the target.

## Quick Start

```bash
cd models_docker
cp .env.example .env
docker compose up --build -d
docker compose logs -f api
```

Health:

```bash
curl -s http://localhost:8000/health
```

The LLM and STT weights are downloaded lazily. `/health` can be `degraded` while llama.cpp is downloading Gemma. STT loads on the first transcription or WebSocket commit.

## Main Endpoints

```text
GET  /health
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
POST /v1/audio/speech
GET  /v1/audio/voices
POST /v1/audio/transcriptions
POST /v1/audio/vad
POST /v1/audio/turn
WS   /v1/audio/stream
POST /v1/voice/chat
```

The WebSocket endpoint accepts binary PCM16 mono audio at 16 kHz. It emits JSON events for readiness, VAD state changes, and final transcripts.

## Examples

Chat:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "content-type: application/json" \
  -H "authorization: Bearer $API_KEY" \
  -d '{"model":"gemma-3-12b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

TTS:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "content-type: application/json" \
  -H "authorization: Bearer $API_KEY" \
  -d '{"model":"kokoro","input":"Hello from Kokoro.","voice":"am_michael","response_format":"wav"}' \
  --output speech.wav
```

STT:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -H "authorization: Bearer $API_KEY" \
  -F model=nemotron-speech-streaming-en-0.6b \
  -F file=@sample.wav
```

VAD:

```bash
curl http://localhost:8000/v1/audio/vad \
  -H "authorization: Bearer $API_KEY" \
  -F file=@sample.wav
```

Smart Turn:

```bash
curl http://localhost:8000/v1/audio/turn \
  -H "authorization: Bearer $API_KEY" \
  -F file=@sample.wav
```

## Useful Environment Variables

| Variable | Default | Notes |
| --- | --- | --- |
| `API_KEY` | empty | If empty, auth is disabled. Set it in production. |
| `LLM_MODEL_NAME` | `gemma-3-12b` | Model alias exposed by llama.cpp. |
| `LLM_HF_REPO` | `google/gemma-3-12b-it-qat-q4_0-gguf` | GGUF repo for llama.cpp `-hf`. |
| `LLM_HF_QUANT` | `Q4_0` | Gemma 3 QAT quant. |
| `LLM_CONTEXT` | `8192` | Reduce if VRAM is tight. |
| `LLM_PARALLEL` | `2` | llama.cpp parallel slots. |
| `LLM_GPU_LAYERS` | `99` | Keep model layers on GPU. |
| `STT_MODEL_ID` | `nvidia/nemotron-speech-streaming-en-0.6b` | NeMo/HF model id. |
| `STT_DEVICE` | `cuda` | Set `cpu` only for debugging. |
| `TTS_DEVICE` | `cuda` | Set `cpu` to leave more GPU headroom for LLM/STT. |
| `TTS_DEFAULT_SPEAKER` | `am_michael` | Must be a Kokoro voice id. |
| `VAD_CONFIDENCE` | `0.7` | Silero speech confidence threshold. |
| `TURN_CPU_COUNT` | `1` | ONNX CPU threads for Smart Turn. |

## Notes

- `llm` uses the official CUDA 12 llama.cpp server image, not a local source build.
- `api` keeps CUDA 12.4 through the PyTorch base image.
- `espeak-ng` is installed in the API image for Kokoro G2P.
- `soundfile` handles WAV/FLAC/OGG uploads; ffmpeg is used as a fallback decoder for other containers.
