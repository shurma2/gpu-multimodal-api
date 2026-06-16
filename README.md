# GPU Multimodal API — Gemma 4 12B + Qwen3-TTS in one container

Run two models in parallel behind a single OpenAI-compatible API:

| Model | Role | Engine | Endpoints |
|-------|------|--------|-----------|
| `google/gemma-4-12B-it-qat-q4_0-gguf` (Q4_0) | Chat + **vision** + **audio-in** | llama.cpp `llama-server` | `/v1/chat/completions`, `/v1/completions` |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | **Streaming TTS**, fluent English | PyTorch | `/v1/audio/speech`, `/v1/audio/voices` |

One image, three processes managed by `supervisord`:

```
                       :8000 (public, OpenAI-compatible, API-key auth)
 client ───────────►  ┌──────────────────────────────────────────────┐
                      │            FastAPI gateway                     │
                      │   /v1/chat/completions   /v1/audio/speech      │
                      │   /v1/voice/chat         /v1/models  /health   │
                      └───────┬───────────────────────────┬───────────┘
              127.0.0.1:8081  │                            │  127.0.0.1:8082
                      ┌───────▼────────┐          ┌────────▼─────────┐
                      │  llama-server  │          │  Qwen3-TTS svc   │
                      │   (Gemma 4)    │          │   (FastAPI)      │
                      └────────────────┘          └──────────────────┘
                                  shared single NVIDIA GPU
```

## Quick start

```bash
cp .env.example .env
#  - set API_KEY
#  - set HF_TOKEN  (Gemma is gated: accept its license on Hugging Face first)

docker compose up --build -d
docker compose logs -f          # first boot downloads ~9 GB of weights

curl -s localhost:8000/health   # {"status":"ok","services":{"llm":"ok","tts":"ok"}}
```

Weights are cached in `./models`, so subsequent starts are fast.

> Requires the **NVIDIA Container Toolkit** on the host (`--gpus all`). The
> compose file already requests all GPUs.

## VRAM — will it fit?

Both models share one GPU. Rough budget at defaults (8K context, Q8 KV cache):

| Component | VRAM |
|-----------|------|
| Gemma 4 12B Q4_0 weights | ~7.0 GB |
| KV cache (8K ctx, q8_0) + mm buffers | ~2–3 GB |
| Qwen3-TTS 0.6B (bf16) + runtime | ~2 GB |
| **Total** | **~11–12 GB** |

| GPU | VRAM | Verdict |
|-----|------|---------|
| RTX A4000 | 16 GB | ✅ Comfortable — can raise `LLM_CONTEXT` / `LLM_PARALLEL` |
| RTX 4070 Ti | 12 GB | ✅ OK — keep defaults, or `LLM_CONTEXT=4096` |
| RTX 3060 | 12 GB | ✅ OK (slower) — `LLM_CONTEXT=4096` |
| RTX 4060 | 8 GB | ⚠️ Both won't fit. Options below. |

**For 8 GB cards** pick one:
- `TTS_DEVICE=cpu` — keep Gemma on GPU, run TTS on CPU (slower TTS, no streaming latency win).
- Use a smaller audio-capable Gemma 4 variant (E2B / E4B) via `LLM_HF_REPO` — check the exact GGUF repo name on Hugging Face.
- Lower `LLM_GPU_LAYERS` to spill some layers to system RAM (slower).

## API

Auth: send `Authorization: Bearer $API_KEY` on every `/v1/*` request.

### Chat (text)
```bash
curl localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-12b","messages":[{"role":"user","content":"Hi!"}],"stream":true}'
```

### Chat with an image
```bash
curl localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-12b","messages":[{"role":"user","content":[
        {"type":"text","text":"What is in this image?"},
        {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,<...>"}}
      ]}]}'
```

### Chat with audio input (Gemma 4 understands speech)
```bash
curl localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-12b","messages":[{"role":"user","content":[
        {"type":"text","text":"Transcribe and answer."},
        {"type":"input_audio","input_audio":{"data":"<base64-wav>","format":"wav"}}
      ]}]}'
```

### Text-to-speech (full file)
```bash
curl localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts","input":"Hello, how are you today?","voice":"Ryan","response_format":"mp3"}' \
  --output out.mp3
```

### Streaming TTS (low latency)
```bash
curl -N localhost:8000/v1/audio/speech \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts","input":"Streaming speech, chunk by chunk.","stream":true,"response_format":"mp3"}' \
  --output stream.mp3
```
`response_format`: `mp3`, `wav`, `pcm`, `opus`, `aac`, `flac`. For lowest
latency use `pcm` (raw 16-bit mono, sample rate in the `X-Sample-Rate` header).

### Voice chat (Gemma reply → spoken)
```bash
curl -N localhost:8000/v1/voice/chat \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me a fun fact."}],"voice":"Ryan"}' \
  --output reply.mp3
# the text reply is also returned URL-encoded in the X-Assistant-Text header
```

### Works with OpenAI SDKs
```python
from openai import OpenAI
c = OpenAI(base_url="http://your-server:8000/v1", api_key="$API_KEY")
print(c.chat.completions.create(model="gemma-4-12b",
      messages=[{"role":"user","content":"Hello"}]).choices[0].message.content)
c.audio.speech.create(model="qwen3-tts", voice="Ryan",
      input="Hello there").stream_to_file("hi.mp3")
```

## Voices

Always-English by default (`TTS_DEFAULT_SPEAKER=Ryan`). Fluent-English options:
**Ryan** (dynamic male) and **Aiden** (sunny American male). Any unknown/OpenAI
voice name falls back to the default, so a single consistent voice is guaranteed.
List all: `GET /v1/audio/voices`. Tune delivery with `instruct`
(e.g. `"Speak slowly and warmly"`).

## Configuration

All behaviour is env-driven — see [.env.example](.env.example). Key knobs:
`API_KEY`, `HF_TOKEN`, `LLM_CONTEXT`, `LLM_PARALLEL`, `LLM_GPU_LAYERS`,
`LLM_PERF_ARGS`, `TTS_DEFAULT_SPEAKER`, `TTS_DEVICE`.

## Notes & caveats

- **Bleeding edge.** Gemma 4 and Qwen3-TTS are recent. The image builds
  `llama.cpp` from `master` (`LLAMA_CPP_REF`) for newest model support. If
  `--flash-attn` fails to parse on a very new build, set
  `LLM_PERF_ARGS="--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0"`.
- **Native TTS streaming** is used automatically if the installed `qwen-tts`
  exposes `generate_custom_voice(stream=True)`; otherwise the service falls back
  to full synthesis sliced into frames — the HTTP streaming contract is
  identical either way. See [tts/engine.py](tts/engine.py).
- **Concurrency.** TTS GPU calls are serialised with a lock (one TTS request at
  a time); the LLM handles `LLM_PARALLEL` concurrent slots. Scale TTS by running
  more replicas behind the gateway.
- **GPU build targets** Turing→Ada (`CUDA_ARCHS=75;80;86;89`). Add older archs
  in the Dockerfile if your rented card is older.
