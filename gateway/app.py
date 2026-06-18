"""Unified OpenAI-compatible gateway.

This is the only publicly exposed process. It forwards LLM traffic to the
llama.cpp server (Gemma 4) and TTS traffic to the Qwen3-TTS service, enforces a
single API key, and adds a convenience voice-chat endpoint that pipes Gemma's
reply straight into TTS.

Public endpoints
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions     -> Gemma 4 (text + image + audio in, streaming)
    POST /v1/completions          -> Gemma 4
    POST /v1/audio/speech         -> Qwen3-TTS (streaming via "stream": true)
    GET  /v1/audio/voices         -> Qwen3-TTS voices
    POST /v1/voice/chat           -> Gemma 4 reply, returned as streamed speech
"""

from __future__ import annotations

import json
import os
from urllib.parse import quote
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8081").rstrip("/")
TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://127.0.0.1:8082").rstrip("/")
API_KEY = os.environ.get("API_KEY", "").strip()
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gemma-4-12b")
TTS_MODEL_NAME = os.environ.get("TTS_MODEL_NAME", "qwen3-tts")
TIMEOUT = float(os.environ.get("GATEWAY_TIMEOUT", "600"))

app = FastAPI(title="GPU Multimodal API", version="1.0.0")
client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT, connect=10.0))


@app.on_event("shutdown")
async def _close():
    await client.aclose()


def _auth(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


async def _proxy_stream(method: str, url: str, content: bytes, headers: dict) -> StreamingResponse:
    """Forward a request and stream the response body back verbatim (works for
    both SSE chat streams and binary audio)."""
    req = client.build_request(method, url, content=content, headers=headers)
    resp = await client.send(req, stream=True)

    async def body():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    passthrough = {
        k: v for k, v in resp.headers.items()
        if k.lower() in ("x-sample-rate", "x-voice")
    }
    return StreamingResponse(
        body(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
        headers=passthrough,
    )


@app.get("/health")
async def health():
    services = {}
    for name, base in (("llm", LLM_BASE_URL), ("tts", TTS_BASE_URL)):
        try:
            r = await client.get(f"{base}/health", timeout=5.0)
            services[name] = "ok" if r.status_code == 200 else f"down({r.status_code})"
        except Exception:
            services[name] = "down"
    healthy = all(v == "ok" for v in services.values())
    return JSONResponse(
        {"status": "ok" if healthy else "degraded", "services": services},
        status_code=200 if healthy else 503,
    )


@app.get("/debug/logs")
async def debug_logs(authorization: Optional[str] = Header(None), lines: int = 80):
    """Tail the llama/tts process logs (written to /models/*.log) over HTTP, so
    the backends can be diagnosed without shell access to the container."""
    _auth(authorization)
    out = {}
    for name, path in (("llama", "/models/llama.log"), ("tts", "/models/tts.log")):
        try:
            with open(path, "r", errors="replace") as f:
                out[name] = f.read().splitlines()[-int(lines):]
        except FileNotFoundError:
            out[name] = ["<no log yet>"]
        except Exception as e:  # noqa: BLE001
            out[name] = [f"<error reading log: {e}>"]
    return out


@app.get("/v1/models")
async def models(authorization: Optional[str] = Header(None)):
    _auth(authorization)
    return {
        "object": "list",
        "data": [
            {"id": LLM_MODEL_NAME, "object": "model", "owned_by": "google",
             "capabilities": ["chat", "text", "image", "audio"]},
            {"id": TTS_MODEL_NAME, "object": "model", "owned_by": "qwen",
             "capabilities": ["tts", "streaming"]},
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    body = await request.body()
    return await _proxy_stream(
        "POST", f"{LLM_BASE_URL}/v1/chat/completions", body, {"content-type": "application/json"}
    )


@app.post("/v1/completions")
async def completions(request: Request, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    body = await request.body()
    return await _proxy_stream(
        "POST", f"{LLM_BASE_URL}/v1/completions", body, {"content-type": "application/json"}
    )


@app.post("/v1/audio/speech")
async def audio_speech(request: Request, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    body = await request.body()
    return await _proxy_stream(
        "POST", f"{TTS_BASE_URL}/v1/audio/speech", body, {"content-type": "application/json"}
    )


@app.get("/v1/audio/voices")
async def audio_voices(authorization: Optional[str] = Header(None)):
    _auth(authorization)
    r = await client.get(f"{TTS_BASE_URL}/v1/audio/voices")
    return JSONResponse(r.json(), status_code=r.status_code)


@app.post("/v1/voice/chat")
async def voice_chat(request: Request, authorization: Optional[str] = Header(None)):
    """One-shot voice agent: send chat `messages` (text, image, or audio parts),
    get Gemma's reply synthesised to speech and streamed back. The text reply is
    also returned URL-encoded in the `X-Assistant-Text` header."""
    _auth(authorization)
    payload = await request.json()
    messages = payload.get("messages")
    if not messages:
        raise HTTPException(400, "missing 'messages'")

    llm_req = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "stream": False,
        "temperature": payload.get("temperature", 0.7),
        "max_tokens": payload.get("max_tokens", 512),
    }
    r = await client.post(f"{LLM_BASE_URL}/v1/chat/completions", json=llm_req)
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"llm error: {r.text[:500]}")
    text = r.json()["choices"][0]["message"]["content"]

    tts_req = json.dumps({
        "model": TTS_MODEL_NAME,
        "input": text,
        "voice": payload.get("voice"),
        "language": payload.get("language"),
        "instruct": payload.get("instruct"),
        "response_format": payload.get("response_format", "mp3"),
        "stream": payload.get("stream", True),
    }).encode()

    resp = await _proxy_stream(
        "POST", f"{TTS_BASE_URL}/v1/audio/speech", tts_req, {"content-type": "application/json"}
    )
    resp.headers["X-Assistant-Text"] = quote(text[:2000])
    return resp
