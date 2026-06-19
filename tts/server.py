"""HTTP service around the Qwen3-TTS engine.

Exposes an OpenAI-compatible text-to-speech endpoint plus streaming. It binds to
127.0.0.1 only; the public surface is the gateway. Auth is handled by the
gateway, so this service trusts its caller.

Endpoints
    GET  /health
    GET  /v1/models
    GET  /v1/audio/voices
    POST /v1/audio/speech   (OpenAI-compatible, +streaming via "stream": true)
"""

from __future__ import annotations

import io
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from typing import Iterator, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .engine import ENGLISH_VOICES, SPEAKERS, TTSEngine

TTS_MODEL_NAME = os.environ.get("TTS_MODEL_NAME", "qwen3-tts")

# response_format -> (ffmpeg output args or None for native, media type)
_FORMATS = {
    "wav": (None, "audio/wav"),
    "pcm": (None, "audio/L16"),
    "flac": (["-f", "flac"], "audio/flac"),
    "mp3": (["-f", "mp3", "-b:a", "128k"], "audio/mpeg"),
    "opus": (["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"], "audio/ogg"),
    "aac": (["-f", "adts", "-c:a", "aac", "-b:a", "128k"], "audio/aac"),
}

engine: Optional[TTSEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = TTSEngine()
    yield


app = FastAPI(title="Qwen3-TTS service", version="1.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# encoding helpers
# --------------------------------------------------------------------------- #
def _to_s16le(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _encode_full(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    if fmt == "pcm":
        return _to_s16le(audio)
    if fmt in ("wav", "flac"):
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format=fmt.upper(), subtype="PCM_16" if fmt == "wav" else None)
        return buf.getvalue()
    # compressed formats via ffmpeg (one-shot)
    out_args, _ = _FORMATS[fmt]
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0", *out_args, "pipe:1"],
        input=_to_s16le(audio), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return proc.stdout


def _wav_header(sr: int, channels: int = 1, bits: int = 16) -> bytes:
    """A streaming WAV header with 'unknown' (max) sizes, so players can start
    immediately without knowing the total length up front."""
    import struct

    byte_rate = sr * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sr, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


def _stream_native(chunks: Iterator[np.ndarray], sr: int, fmt: str) -> Iterator[bytes]:
    """pcm / wav: no transcoding, lowest latency."""
    if fmt == "wav":
        yield _wav_header(sr)
    for chunk in chunks:
        if chunk.size:
            yield _to_s16le(chunk)


def _stream_ffmpeg(chunks: Iterator[np.ndarray], sr: int, fmt: str) -> Iterator[bytes]:
    """Compressed formats: pipe s16le frames through ffmpeg as they arrive."""
    out_args, _ = _FORMATS[fmt]
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error",
         "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0", *out_args, "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )

    def _feed():
        try:
            for chunk in chunks:
                if chunk.size:
                    proc.stdin.write(_to_s16le(chunk))
        except Exception:
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    threading.Thread(target=_feed, daemon=True).start()
    try:
        while True:
            data = proc.stdout.read(8192)
            if not data:
                break
            yield data
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/health")
async def health():
    ready = engine is not None and engine.sample_rate is not None
    return JSONResponse({"status": "ok" if ready else "loading"}, status_code=200 if ready else 503)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": TTS_MODEL_NAME, "object": "model", "owned_by": "qwen"}]}


@app.get("/v1/audio/voices")
async def voices():
    return {
        "default": engine.default_speaker if engine else None,
        "language": engine.default_language if engine else None,
        "voices": sorted(SPEAKERS),
        "english_voices": ENGLISH_VOICES,
    }


@app.post("/v1/audio/speech")
async def speech(request: Request):
    if engine is None:
        raise HTTPException(503, "engine not ready")

    body = await request.json()
    text = body.get("input") or body.get("text")
    if not text:
        raise HTTPException(400, "missing 'input'")

    fmt = (body.get("response_format") or "mp3").lower()
    if fmt not in _FORMATS:
        raise HTTPException(400, f"unsupported response_format '{fmt}'")

    speaker = engine.resolve_speaker(body.get("voice"))
    language = body.get("language")
    instruct = body.get("instruct")
    stream = bool(body.get("stream")) or body.get("stream_format") == "audio"

    _, media_type = _FORMATS[fmt]
    headers = {"X-Sample-Rate": str(engine.sample_rate), "X-Voice": speaker}

    if stream:
        chunks = engine.stream(text, language=language, speaker=speaker, instruct=instruct)
        encoder = _stream_native if fmt in ("wav", "pcm") else _stream_ffmpeg
        return StreamingResponse(
            encoder(chunks, engine.sample_rate, fmt),
            media_type=media_type,
            headers=headers,
        )

    audio, sr = engine.synth(text, language=language, speaker=speaker, instruct=instruct)
    data = _encode_full(audio, sr, fmt)
    return Response(content=data, media_type=media_type, headers=headers)
