# Speaker Voice API — Technical Reference

Self-contained voice-agent backend for a single NVIDIA GPU host (target:
RTX 3090 Ti, Ampere `sm_86`, CUDA 12.x driver). One Docker image runs the whole
stack: LLM, STT, VAD, end-of-turn detection and TTS behind one FastAPI gateway
on port `8000`.

```
                          ┌─────────────────────────── container ───────────────────────────┐
client ──HTTP/WS──► :8000 │  FastAPI gateway (uvicorn)                                        │
                          │    ├─ /v1/chat,/v1/completions ─proxy─► llama-server :8081 (GPU)  │
                          │    ├─ /v1/audio/speech ──────────────► Kokoro TTS   (in-process)  │
                          │    ├─ /v1/audio/transcriptions ──────► Nemotron STT (in-process)  │
                          │    ├─ /v1/audio/vad,/turn ───────────► Silero + Smart Turn (CPU)  │
                          │    └─ /v1/audio/stream (WebSocket) ──► VAD + turn + STT combined  │
                          │  supervisord runs: [llama] + [gateway]                            │
                          └───────────────────────────────────────────────────────────────────┘
```

## Models

| Role         | Model                                          | Runtime                    |
| ------------ | ---------------------------------------------- | -------------------------- |
| LLM          | `google/gemma-4-12B-it-qat-q4_0-gguf` (Q4_0)   | llama.cpp server, GPU      |
| STT          | `nvidia/nemotron-speech-streaming-en-0.6b`     | NVIDIA NeMo, GPU           |
| VAD          | Silero VAD ONNX (bundled with Pipecat)         | CPU                        |
| End of turn  | Smart Turn v3 ONNX (bundled with Pipecat)      | CPU                        |
| TTS          | Kokoro-82M, 24 kHz                             | GPU (CPU optional)         |

Gemma 4 is multimodal, but the LLM is served **text-only** via `--no-mmproj`
(skips the multimodal projector, saving VRAM). Gemma is a **gated** HF repo, so
a valid `HF_TOKEN` that has accepted the license is required to download it.

## Image layout (`Dockerfile`)

Two-stage build:

1. **`llama-builder`** — `nvidia/cuda:12.4.1-devel-ubuntu22.04`. Clones a pinned
   llama.cpp release (`LLAMA_CPP_REF`, must be ≥ `b9724` for Gemma 4 support)
   and builds `llama-server` with `-DGGML_CUDA=ON` for `sm_86`. The binary, the
   ggml `.so` files **and the CUDA runtime libraries** are staged under
   `/opt/llama`.
2. **`runtime`** — `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`. Adds
   espeak-ng/ffmpeg, installs the Python deps (`cu124` wheels), copies
   `/opt/llama` and the app, and runs everything under `supervisord`.

### Why the CUDA runtime libs are copied explicitly

`llama-server` is built against `libcudart.so.12` / `libcublas.so.12` /
`libcublasLt.so.12`, which live in `/usr/local/cuda/lib64` of the *devel* image.
The runtime stage is the PyTorch image, where CUDA ships inside a conda path
`llama-server` cannot see. With only ggml's own `.so` files copied, the binary
crash-looped with:

```
/opt/llama/bin/llama-server: error while loading shared libraries:
libcudart.so.12: cannot open shared object file: No such file or directory
```

The build now copies those three CUDA libraries from the builder into
`/opt/llama/lib`, and `entrypoint.sh` prepends that directory to
`LD_LIBRARY_PATH` **only for the `llama-server` process** — so they never shadow
the CUDA libraries PyTorch (STT/TTS) loads. `libcuda.so` (the driver) is *not*
copied; the NVIDIA Container Toolkit injects it from the host at run time, so
the container must always be started with GPU access.

## Process model

`supervisord.conf` supervises two long-running programs:

- **`llama`** → `entrypoint.sh llama` → `llama-server` on `127.0.0.1:8081`,
  logs to `/models/llama.log`.
- **`gateway`** → uvicorn on `0.0.0.0:8000`, logs to `/models/api.log`.

Both logs are exposed through `GET /debug/logs?lines=N` (auth required), which
is the primary way to inspect a rented host without shell access.

`/models` is a volume holding the HF cache (`/models/hf-cache`), the NeMo cache
(`/models/nemo-cache`) and the logs, so weights download once and persist
across restarts.

## The unified streaming channel — `WS /v1/audio/stream`

VAD, end-of-turn detection and STT are bound into **one** WebSocket so a
consumer never has to correlate three separate signals. The client streams mic
audio and listens on the same socket for a single decisive `end_of_turn` event.

**Audio format:** raw **PCM16, mono, little-endian, 16 kHz** binary frames
(announced in the `ready` event). No container/header — just samples.

**Client → server**

| Frame            | Meaning                                              |
| ---------------- | ---------------------------------------------------- |
| binary           | a chunk of PCM16 audio                               |
| `{"type":"commit"}` | force end-of-turn now (transcribe the buffer)     |
| `{"type":"reset"}`  | drop the buffered utterance, start fresh          |
| `{"type":"ping"}`   | liveness/readiness check                          |

**Server → client** (all JSON on the one channel)

| Event             | Payload                                                       |
| ----------------- | ------------------------------------------------------------- |
| `ready`           | `sample_rate`, `encoding`, `models:{vad,turn,stt}`            |
| `speech_started`  | VAD detected the user started talking                         |
| `speech_stopped`  | VAD detected the user went quiet                              |
| `end_of_turn`     | `text` (final transcript), `complete:true`, `probability`, `processing_ms`, `reason` (`vad`\|`commit`) |
| `pong`            | reply to `ping`: `stt` load state, `speaking` flag            |
| `error`           | `error` message                                               |

The flow: VAD gates which audio is buffered and emits start/stop; Smart Turn v3
decides when the **turn** is actually finished (not just a pause); on a complete
turn the buffered utterance is transcribed by Nemotron and shipped as
`end_of_turn`. **Receiving `end_of_turn` is the signal to stop listening** and
hand `text` to the LLM. Nemotron runs as a batch transcription over the
buffered turn, so there are no interim partials — only the final text.

Minimal client:

```python
import asyncio, json, websockets

async def main():
    url = "ws://HOST:8000/v1/audio/stream?api_key=API_KEY"
    async with websockets.connect(url) as ws:
        print(json.loads(await ws.recv()))      # {"type":"ready", ...}
        async def reader():
            async for msg in ws:
                e = json.loads(msg)
                if e["type"] == "end_of_turn":
                    print("USER SAID:", e["text"])  # stop listening, call LLM
        asyncio.create_task(reader())
        for chunk in pcm16_chunks_from_mic():        # 16 kHz mono PCM16
            await ws.send(chunk)

asyncio.run(main())
```

## HTTP endpoints

```
GET  /health                     status + per-service state (no auth)
GET  /debug/logs?lines=N         tail llama/gateway logs (auth)
GET  /v1/models                  list served models (auth)
POST /v1/chat/completions        OpenAI-compatible, streams (auth)
POST /v1/completions             OpenAI-compatible (auth)
POST /v1/audio/speech            Kokoro TTS, OpenAI-compatible (auth)
GET  /v1/audio/voices            list Kokoro voices (auth)
POST /v1/audio/transcriptions    Nemotron STT, multipart file (auth)
POST /v1/audio/vad               Silero VAD over an uploaded file (auth)
POST /v1/audio/turn              Smart Turn over an uploaded file (auth)
WS   /v1/audio/stream            unified VAD + turn + STT channel (auth)
POST /v1/voice/chat              messages -> LLM -> TTS audio (auth)
```

Auth is `Authorization: Bearer <API_KEY>` (or `?api_key=` query param on the
WebSocket). Empty `API_KEY` disables auth.

`/health` reports `ok` only when the LLM answers and STT has no load error;
otherwise `degraded` with a per-service breakdown (`llm`, `tts`, `stt`, `vad`,
`turn`). STT and the LLM load lazily, so `degraded` is normal during first-boot
downloads.

Quick checks:

```bash
curl -s http://HOST:8000/health

curl http://HOST:8000/v1/chat/completions \
  -H "authorization: Bearer $API_KEY" -H "content-type: application/json" \
  -d '{"model":"gemma-4-12b","messages":[{"role":"user","content":"Hi"}],"stream":true}'

curl http://HOST:8000/v1/audio/speech \
  -H "authorization: Bearer $API_KEY" -H "content-type: application/json" \
  -d '{"input":"Hello from Kokoro.","voice":"am_michael","response_format":"wav"}' \
  --output speech.wav

curl http://HOST:8000/v1/audio/transcriptions \
  -H "authorization: Bearer $API_KEY" -F file=@sample.wav
```

## Configuration

All config is environment variables; secrets live in a git-ignored `.env`
(`cp .env.example .env`). Defaults are baked into the `Dockerfile` /
`gateway/config.py`, so `.env` only needs the values you override.

| Variable               | Default                                  | Notes |
| ---------------------- | ---------------------------------------- | ----- |
| `API_KEY`              | empty                                    | Bearer token; empty disables auth |
| `HF_TOKEN`             | empty                                    | **Required** — gated Gemma + STT downloads |
| `LLM_MODEL_NAME`       | `gemma-4-12b`                            | Alias exposed by the API |
| `LLM_HF_REPO`          | `google/gemma-4-12B-it-qat-q4_0-gguf`    | GGUF repo for llama.cpp `-hf` (must be a *-gguf* repo, not *-unquantized*) |
| `LLM_HF_QUANT`         | `Q4_0`                                   | Quant selector |
| `LLM_CONTEXT`          | `8192`                                   | Context window |
| `LLM_PARALLEL`         | `2`                                      | llama.cpp parallel slots |
| `LLM_GPU_LAYERS`       | `99`                                     | Layers on GPU |
| `LLM_PERF_ARGS`        | flash-attn + q8_0 KV cache + `--no-mmproj` | Extra llama-server flags |
| `STT_DEVICE`           | `cuda`                                   | `cpu` only for debugging |
| `TTS_DEVICE`           | `cuda`                                   | `cpu` frees GPU for LLM/STT |
| `TTS_DEFAULT_SPEAKER`  | `am_michael`                             | Kokoro voice id |
| `VAD_CONFIDENCE`       | `0.7`                                    | Silero speech threshold |
| `VAD_START_SECS`/`VAD_STOP_SECS` | `0.2`/`0.2`                    | VAD debounce |
| `TURN_STOP_SECS`       | `3.0`                                    | Silence before turn analysis |
| `TURN_MAX_DURATION_SECS` | `8.0`                                  | Max audio fed to Smart Turn |
| `TURN_CPU_COUNT`       | `1`                                      | ONNX CPU threads for Smart Turn |

## Run

Single container (production / GPU rental — what runs on the host):

```bash
docker run -d --gpus all --env-file .env \
  -p 8000:8000 -v speaker-models:/models \
  reassel/gpu-multimodal-api:latest
```

Or with Compose (`docker-compose.yml`) — runs the official llama.cpp CUDA image
as a separate `llm` service and builds the gateway image:

```bash
cp .env.example .env   # fill in API_KEY + HF_TOKEN
docker compose up --build -d
docker compose logs -f api
```

`--gpus all` (or the Compose `deploy.resources` block) is mandatory: it injects
the host `libcuda.so` the GPU build needs.

## CI / build caching (`.github/workflows/build.yml`)

Every push to `main` builds `linux/amd64` and pushes
`reassel/gpu-multimodal-api:{1.0,latest}` to Docker Hub.

The slow stages are the llama.cpp CUDA build (~15 min) and the torch/NeMo pip
install. GitHub Actions keeps **no** Docker layer cache between runs by default,
so without help every push rebuilt them from scratch. The workflow uses buildx
**registry cache** (`mode=max`) stored in a dedicated `:buildcache` tag on
Docker Hub:

```yaml
cache-from: type=registry,ref=reassel/gpu-multimodal-api:buildcache
cache-to:   type=registry,ref=reassel/gpu-multimodal-api:buildcache,mode=max
```

Because `LLAMA_CPP_REF` and `requirements.txt` change rarely, those heavy layers
are pulled from the registry and only the changed app layers rebuild. A registry
cache is used instead of committing a cache folder to git (multi-GB binaries do
not belong in the repo) and instead of GitHub's `type=gha` cache (its 10 GB cap
is too small for the CUDA + torch layers). To force a clean rebuild, delete the
`:buildcache` tag on Docker Hub.
