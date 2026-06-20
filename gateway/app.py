"""Unified voice API.

Public surface:
    GET  /health
    GET  /v1/models
    POST /v1/chat/completions       -> llama.cpp OpenAI-compatible proxy
    POST /v1/completions            -> llama.cpp OpenAI-compatible proxy
    POST /v1/audio/speech           -> Kokoro TTS
    POST /v1/audio/transcriptions   -> Parakeet-EOU STT
    POST /v1/audio/vad              -> Silero VAD ONNX
    POST /v1/audio/turn             -> Pipecat Smart Turn v3.2
    WS   /v1/audio/stream           -> streaming ASR + pause-tolerant end-of-thought
    POST /v1/voice/chat             -> LLM reply synthesized to speech

The WebSocket channel fuses streaming ASR (Parakeet-EOU), Silero VAD pauses and
Smart Turn into one event stream. The client streams raw PCM16 mono 16 kHz audio
(binary frames) and receives JSON events: live `partial_text`, per-utterance
`final_text`, acoustic `speech_pause`, and finally `thought_end` once the user is
confirmed done — the signal to stop listening and act. See README.md.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import quote

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, Response, StreamingResponse

from gateway.config import get_settings
from tts.codec import FORMATS, encode_full, stream_ffmpeg, stream_native
from tts.engine import ENGLISH_VOICES, SPEAKERS, TTSEngine
from voice.audio import AudioDecodeError, pcm16_bytes_to_float
from voice.endpoint import EndpointController
from voice.stt import STTEngine
from voice.turn import VADTurnService

settings = get_settings()

client: Optional[httpx.AsyncClient] = None
tts_engine: Optional[TTSEngine] = None
stt_engine: Optional[STTEngine] = None
vad_turn_service: Optional[VADTurnService] = None
tts_error: Optional[str] = None

tts_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.gateway_timeout, connect=10.0),
    )
    yield
    await client.aclose()


app = FastAPI(title="Speaker Voice API", version="2.0.0", lifespan=lifespan)


def _client() -> httpx.AsyncClient:
    if client is None:
        raise HTTPException(503, "HTTP client is not ready")
    return client


def _auth(authorization: Optional[str]) -> None:
    if not settings.api_key:
        return
    if authorization != f"Bearer {settings.api_key}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


async def _auth_ws(websocket: WebSocket) -> bool:
    if not settings.api_key:
        return True
    authorization = websocket.headers.get("authorization")
    api_key = websocket.query_params.get("api_key")
    if authorization == f"Bearer {settings.api_key}" or api_key == settings.api_key:
        return True
    await websocket.close(code=1008)
    return False


async def _proxy_stream(method: str, url: str, content: bytes, headers: dict) -> StreamingResponse:
    req = _client().build_request(method, url, content=content, headers=headers)
    resp = await _client().send(req, stream=True)

    async def body():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    passthrough = {
        k: v
        for k, v in resp.headers.items()
        if k.lower() in ("x-sample-rate", "x-voice")
    }
    return StreamingResponse(
        body(),
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type"),
        headers=passthrough,
    )


async def get_tts() -> TTSEngine:
    global tts_engine, tts_error
    if tts_engine is not None:
        return tts_engine
    async with tts_lock:
        if tts_engine is not None:
            return tts_engine
        os.environ["TTS_DEVICE"] = settings.tts_device
        os.environ["TTS_DEFAULT_SPEAKER"] = settings.tts_default_speaker
        os.environ["TTS_DEFAULT_LANGUAGE"] = settings.tts_default_language
        try:
            tts_engine = await asyncio.to_thread(TTSEngine)
            tts_error = None
            return tts_engine
        except Exception as exc:
            tts_error = str(exc)
            raise


def get_stt() -> STTEngine:
    global stt_engine
    if stt_engine is None:
        stt_engine = STTEngine(settings)
    return stt_engine


def get_vad_turn() -> VADTurnService:
    global vad_turn_service
    if vad_turn_service is None:
        vad_turn_service = VADTurnService(settings)
    return vad_turn_service


async def _check_llm() -> str:
    base = settings.llm_base_url.rstrip("/")
    try:
        r = await _client().get(f"{base}/health", timeout=5.0)
        if r.status_code == 200:
            return "ok"
        if r.status_code != 404:
            return f"down({r.status_code})"
    except Exception:
        pass
    try:
        r = await _client().get(f"{base}/v1/models", timeout=5.0)
        return "ok" if r.status_code == 200 else f"down({r.status_code})"
    except Exception:
        return "down"


@app.get("/health")
async def health():
    services = {
        "llm": await _check_llm(),
        "tts": "ok" if tts_engine is not None else (f"error: {tts_error}" if tts_error else "not_loaded"),
        "stt": get_stt().status,
        "vad": "ok",
        "turn": "ok",
    }
    healthy = services["llm"] == "ok" and not str(services["stt"]).startswith("error")
    return JSONResponse(
        {"status": "ok" if healthy else "degraded", "services": services},
        status_code=200 if healthy else 503,
    )


@app.get("/debug/logs")
async def debug_logs(authorization: Optional[str] = Header(None), lines: int = 80):
    _auth(authorization)
    out = {}
    for name, path in (
        ("api", "/models/api.log"),
        ("llm", "/models/llama.log"),
    ):
        try:
            with open(path, "r", errors="replace") as f:
                out[name] = f.read().splitlines()[-int(lines):]
        except FileNotFoundError:
            out[name] = ["<no log file in this container>"]
        except Exception as exc:
            out[name] = [f"<error reading log: {exc}>"]
    return out


@app.get("/v1/models")
async def models(authorization: Optional[str] = Header(None)):
    _auth(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": settings.llm_model_name,
                "object": "model",
                "owned_by": "google",
                "capabilities": ["chat", "text", "streaming"],
            },
            {
                "id": settings.stt_model_name,
                "object": "model",
                "owned_by": "nvidia",
                "capabilities": ["stt", "streaming", "fastconformer-rnnt"],
            },
            {
                "id": settings.vad_model_name,
                "object": "model",
                "owned_by": "snakers4",
                "capabilities": ["vad", "onnx", "cpu"],
            },
            {
                "id": settings.turn_model_name,
                "object": "model",
                "owned_by": "pipecat-ai",
                "capabilities": ["turn-detection", "native-audio", "onnx", "cpu"],
            },
            {
                "id": settings.tts_model_name,
                "object": "model",
                "owned_by": "hexgrad",
                "capabilities": ["tts", "streaming", "24khz"],
            },
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    body = await request.body()
    return await _proxy_stream(
        "POST",
        f"{settings.llm_base_url.rstrip('/')}/v1/chat/completions",
        body,
        {"content-type": "application/json"},
    )


@app.post("/v1/completions")
async def completions(request: Request, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    body = await request.body()
    return await _proxy_stream(
        "POST",
        f"{settings.llm_base_url.rstrip('/')}/v1/completions",
        body,
        {"content-type": "application/json"},
    )


@app.post("/v1/audio/speech")
async def audio_speech(request: Request, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    engine = await get_tts()
    body = await request.json()
    text = body.get("input") or body.get("text")
    if not text:
        raise HTTPException(400, "missing 'input'")

    fmt = (body.get("response_format") or "mp3").lower()
    if fmt not in FORMATS:
        raise HTTPException(400, f"unsupported response_format '{fmt}'")

    speaker = engine.resolve_speaker(body.get("voice"))
    language = body.get("language")
    instruct = body.get("instruct")
    stream = bool(body.get("stream")) or body.get("stream_format") == "audio"

    _, media_type = FORMATS[fmt]
    headers = {"X-Sample-Rate": str(engine.sample_rate), "X-Voice": speaker}

    if stream:
        chunks = engine.stream(text, language=language, speaker=speaker, instruct=instruct)
        encoder = stream_native if fmt in ("wav", "pcm") else stream_ffmpeg
        return StreamingResponse(
            encoder(chunks, engine.sample_rate, fmt),
            media_type=media_type,
            headers=headers,
        )

    audio, sr = await asyncio.to_thread(
        engine.synth,
        text,
        language,
        speaker,
        instruct,
    )
    data = await asyncio.to_thread(encode_full, audio, sr, fmt)
    return Response(content=data, media_type=media_type, headers=headers)


@app.get("/v1/audio/voices")
async def audio_voices(authorization: Optional[str] = Header(None)):
    _auth(authorization)
    return {
        "default": settings.tts_default_speaker,
        "language": settings.tts_default_language,
        "voices": sorted(SPEAKERS),
        "english_voices": ENGLISH_VOICES,
    }


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    authorization: Optional[str] = Header(None),
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
):
    _auth(authorization)
    if model and model not in (settings.stt_model_name, settings.stt_model_id):
        raise HTTPException(400, f"unsupported transcription model '{model}'")
    data = await file.read()
    try:
        text = await get_stt().transcribe_bytes(data)
    except AudioDecodeError as exc:
        raise HTTPException(400, str(exc)) from exc
    if response_format == "text":
        return Response(content=text, media_type="text/plain")
    return {"text": text, "model": settings.stt_model_name}


@app.post("/v1/audio/vad")
async def audio_vad(
    authorization: Optional[str] = Header(None),
    file: UploadFile = File(...),
):
    _auth(authorization)
    data = await file.read()
    try:
        return await get_vad_turn().analyze_vad_bytes(data)
    except AudioDecodeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/v1/audio/turn")
async def audio_turn(
    authorization: Optional[str] = Header(None),
    file: UploadFile = File(...),
):
    _auth(authorization)
    data = await file.read()
    try:
        return await get_vad_turn().predict_turn_bytes(data)
    except AudioDecodeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/debug/stt-stream")
async def debug_stt_stream(
    authorization: Optional[str] = Header(None),
    file: UploadFile = File(...),
    ws_chunk_ms: int = Form(default=100),
):
    """Validate the cache-aware streaming STT path over an uploaded clip and
    compare with batch. Returns introspection, per-step timings, streamed vs
    batch text, and any traceback — so the streaming rework can be verified
    remotely without SSH."""
    _auth(authorization)
    data = await file.read()
    try:
        return await get_stt().stream_probe(data, ws_chunk_ms=ws_chunk_ms)
    except AudioDecodeError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.websocket("/v1/audio/stream")
async def audio_stream(websocket: WebSocket):
    """Unified streaming ASR + pause-tolerant end-of-thought channel.

    Three signals are fused (see voice/endpoint.py): Parakeet-EOU streams text
    and emits `<EOU>`/`<EOB>` markers, Silero VAD provides acoustic pauses, and
    Smart Turn semantically confirms an EOU before we declare the thought done.

    Client -> server:
        - binary frames: raw PCM16 mono little-endian at `sample_rate` (16 kHz)
        - text frames (JSON): {"type": "commit"} force thought_end now,
                              {"type": "reset"}  drop the in-progress thought,
                              {"type": "ping"}   liveness/readiness check

    Server -> client (all JSON, single channel):
        - {"type": "ready", ...}              handshake with models + audio format
        - {"type": "partial_text", "text": ..}  live hypothesis (tokens stripped)
        - {"type": "speech_pause"}            Silero went quiet after speech (acoustic)
        - {"type": "final_text", "text": ..}   one completed <EOU> segment
              (several may precede a single thought_end; accumulate them)
        - {"type": "thought_end", "text": .., "segments": [...], "reason": ..}
              the user finished; consumer should act on the accumulated text
        - {"type": "pong", "stt": ...} / {"type": "reset"} / {"type": "error", ...}
    """
    if not await _auth_ws(websocket):
        return
    await websocket.accept()

    service = get_vad_turn()
    stt = get_stt()
    sample_rate = settings.vad_sample_rate
    vad = service.create_vad(sample_rate)
    turn = service.create_turn_analyzer(sample_rate)
    speaking = False
    last_partial = ""
    # Per-connection streaming decoder + endpoint controller. The recognizer
    # runs continuously (it self-endpoints, so it must see the trailing pause).
    stt_session = await stt.create_stream_session()
    controller = EndpointController(settings)

    async def fire_thought_end(reason: str) -> None:
        nonlocal speaking, stt_session, last_partial
        # flush the tail of the recognizer so any open segment is finalized
        result = await stt_session.finalize()
        controller.ingest_stream(result)
        for seg in result["finals"]:
            if seg:
                await websocket.send_json({"type": "final_text", "text": seg})
        await websocket.send_json(controller.take_thought_end(reason))
        turn.clear()
        speaking = False
        last_partial = ""
        stt_session = await stt.create_stream_session()

    await websocket.send_json(
        {
            "type": "ready",
            "sample_rate": sample_rate,
            "encoding": "pcm_s16le",
            "models": {
                "vad": settings.vad_model_name,
                "turn": settings.turn_model_name,
                "stt": settings.stt_model_name,
            },
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            chunk = message.get("bytes")
            if chunk is not None:
                if not chunk:
                    continue
                vad_state = await vad.analyze_audio(chunk)
                vad_name = vad_state.name.lower()
                is_speech = vad_name in ("starting", "speaking", "stopping")

                # Recognizer runs on every frame; EOU detection needs the pause.
                result = await stt_session.feed(pcm16_bytes_to_float(chunk))
                controller.ingest_stream(result)
                if result["partial"] and result["partial"] != last_partial:
                    last_partial = result["partial"]
                    await websocket.send_json(
                        {"type": "partial_text", "text": result["partial"]}
                    )
                for seg in result["finals"]:
                    if seg:
                        await websocket.send_json({"type": "final_text", "text": seg})

                turn.append_audio(chunk, is_speech)
                if is_speech:
                    speaking = True
                elif speaking and vad_name == "quiet":
                    # acoustic pause: surface it and ask Smart Turn to (dis)confirm
                    speaking = False
                    last_partial = ""
                    await websocket.send_json({"type": "speech_pause"})
                    if turn.speech_triggered:
                        turn_state, _ = await turn.analyze_end_of_turn()
                        controller.on_smart_turn(turn_state.name.lower() == "complete")

                if controller.should_fire():
                    await fire_thought_end("eou+turn")
                continue

            text_message = message.get("text")
            if text_message is None:
                continue
            try:
                command = json.loads(text_message)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "error": "text messages must be JSON"})
                continue

            action = command.get("type") or command.get("action")
            if action == "reset":
                turn.clear()
                speaking = False
                last_partial = ""
                stt_session = await stt.create_stream_session()
                controller.reset()
                await websocket.send_json({"type": "reset"})
            elif action in ("commit", "flush"):
                await fire_thought_end("commit")
            elif action == "ping":
                await websocket.send_json(
                    {"type": "pong", "stt": get_stt().status, "speaking": speaking}
                )
            else:
                await websocket.send_json({"type": "error", "error": f"unknown action '{action}'"})
    finally:
        try:
            turn.clear()
        except Exception:
            pass


@app.post("/v1/voice/chat")
async def voice_chat(request: Request, authorization: Optional[str] = Header(None)):
    _auth(authorization)
    payload = await request.json()
    messages = payload.get("messages")
    if not messages:
        raise HTTPException(400, "missing 'messages'")

    llm_req = {
        "model": settings.llm_model_name,
        "messages": messages,
        "stream": False,
        "temperature": payload.get("temperature", 0.7),
        "max_tokens": payload.get("max_tokens", 512),
    }
    r = await _client().post(
        f"{settings.llm_base_url.rstrip('/')}/v1/chat/completions",
        json=llm_req,
    )
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"llm error: {r.text[:500]}")
    text = r.json()["choices"][0]["message"]["content"]

    body = {
        "model": settings.tts_model_name,
        "input": text,
        "voice": payload.get("voice"),
        "language": payload.get("language"),
        "instruct": payload.get("instruct"),
        "response_format": payload.get("response_format", "mp3"),
        "stream": payload.get("stream", True),
    }
    response = await audio_speech(
        request=_MemoryJSONRequest(body),
        authorization=authorization,
    )
    response.headers["X-Assistant-Text"] = quote(text[:2000])
    return response


class _MemoryJSONRequest:
    def __init__(self, body: dict) -> None:
        self._body = body

    async def json(self) -> dict:
        return self._body
