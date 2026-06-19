"""NVIDIA NeMo STT wrapper for Nemotron Speech Streaming EN 0.6B."""

from __future__ import annotations

import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
import soundfile as sf

from gateway.config import Settings
import time

from voice.audio import decode_audio_bytes, to_mono_f32


class STTEngine:
    """Lazy, serialized ASR engine.

    The model is loaded on first use so the API can come up while HF downloads
    or gated model setup is still being fixed.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_id = settings.stt_model_id
        self.model_name = settings.stt_model_name
        self.device = settings.stt_device
        self.sample_rate = settings.stt_sample_rate
        self._model: Any = None
        self._load_error: Optional[str] = None
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def status(self) -> str:
        if self._model is not None:
            return "ok"
        if self._load_error:
            return f"error: {self._load_error}"
        return "not_loaded"

    async def load(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            loop = asyncio.get_running_loop()
            try:
                self._model = await loop.run_in_executor(self._executor, self._load_sync)
                self._load_error = None
            except Exception as exc:
                self._load_error = str(exc)
                raise

    def _load_sync(self) -> Any:
        import torch
        import nemo.collections.asr as nemo_asr

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("STT_DEVICE=cuda requested, but torch.cuda.is_available() is false")

        model = nemo_asr.models.ASRModel.from_pretrained(self.model_id)
        model.eval()
        if self.device:
            model = model.to(self.device)
        return model

    async def create_stream_session(self) -> "STTStreamSession":
        """Per-connection cache-aware streaming decoder. Falls back silently
        (session.ok == False) if the model lacks the streaming API; callers
        should then use batch transcribe_audio on the buffered utterance."""
        await self.load()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, STTStreamSession, self)

    async def stream_probe(self, data: bytes, ws_chunk_ms: int = 100) -> dict[str, Any]:
        """Diagnostic: run the real streaming path over `data` and compare with
        batch. Used by the debug endpoint to validate the NeMo streaming API
        remotely (no SSH). Everything runs in the serialized GPU executor."""
        await self.load()
        audio, sr = decode_audio_bytes(data, target_sr=self.sample_rate)
        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                self._executor, self._stream_probe_sync, audio, sr, ws_chunk_ms
            )

    def _stream_probe_sync(self, audio: np.ndarray, sr: int, ws_chunk_ms: int) -> dict[str, Any]:
        import traceback

        out: dict[str, Any] = {"audio_seconds": round(len(audio) / float(sr), 3)}
        session = STTStreamSession(self)
        out["introspection"] = session.introspection
        out["ok"] = session.ok
        out["step_samples"] = session.step_samples
        if not session.ok:
            out["error"] = session.error
        else:
            ws = max(1, int(sr * ws_chunk_ms / 1000))
            pending = np.zeros(0, dtype=np.float32)
            steps: list[dict[str, Any]] = []
            t_stream = time.time()
            try:
                for i in range(0, len(audio), ws):
                    pending = np.concatenate([pending, audio[i : i + ws]])
                    while pending.shape[0] >= session.step_samples:
                        block = pending[: session.step_samples]
                        pending = pending[session.step_samples :]
                        t0 = time.time()
                        txt = session.step_sync(block, is_last=False)
                        steps.append({"samples": int(block.shape[0]),
                                      "ms": round((time.time() - t0) * 1000, 1),
                                      "text": txt})
                    if len(steps) >= 200:
                        break
                t0 = time.time()
                final = session.step_sync(
                    pending if pending.shape[0] else np.zeros(1, dtype=np.float32),
                    is_last=True,
                )
                steps.append({"samples": int(pending.shape[0]), "final": True,
                              "ms": round((time.time() - t0) * 1000, 1), "text": final})
                out["stream_ms"] = round((time.time() - t_stream) * 1000, 1)
                out["streamed_text"] = final
                out["steps"] = steps
            except Exception as exc:  # noqa: BLE001
                out["ok"] = False
                out["error"] = f"step: {type(exc).__name__}: {exc}"
                out["traceback"] = traceback.format_exc()

        t0 = time.time()
        try:
            out["batch_text"] = self._transcribe_audio_sync(audio, sr)
            out["batch_ms"] = round((time.time() - t0) * 1000, 1)
            out["match"] = (
                str(out.get("streamed_text", "")).strip().lower()
                == str(out["batch_text"]).strip().lower()
            )
        except Exception as exc:  # noqa: BLE001
            out["batch_error"] = f"{type(exc).__name__}: {exc}"
        return out

    async def transcribe_bytes(self, data: bytes) -> str:
        audio, sr = decode_audio_bytes(data, target_sr=self.sample_rate)
        return await self.transcribe_audio(audio, sr)

    async def transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        await self.load()
        audio = to_mono_f32(audio)
        if audio.size == 0:
            return ""

        loop = asyncio.get_running_loop()
        async with self._lock:
            return await loop.run_in_executor(
                self._executor,
                self._transcribe_audio_sync,
                audio,
                sample_rate,
            )

    def _transcribe_audio_sync(self, audio: np.ndarray, sample_rate: int) -> str:
        import torch

        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(path, audio, sample_rate, subtype="PCM_16")
            with torch.inference_mode():
                result = self._model.transcribe([path], batch_size=1)
            return self._extract_text(result)
        finally:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def _extract_text(self, result: Any) -> str:
        item = result
        if isinstance(item, (list, tuple)) and item:
            item = item[0]
        if isinstance(item, str):
            return item
        if hasattr(item, "text"):
            return str(item.text)
        if isinstance(item, dict):
            for key in ("text", "transcript", "prediction"):
                if key in item:
                    return str(item[key])
        return "" if item is None else str(item)


class STTStreamSession:
    """Cache-aware streaming decode state for one connection / one turn.

    Audio is fed incrementally as it arrives (concurrent with VAD + turn), each
    block decoded via `conformer_stream_step` while keeping the encoder cache,
    so the transcript is already done when the turn ends — `finalize()` only
    flushes the tail. GPU work is serialized through the engine's single-worker
    executor + lock to coexist with any batch transcribe.

    If the loaded model does not expose the streaming API, `ok` stays False and
    callers fall back to batch transcription.
    """

    def __init__(self, engine: "STTEngine") -> None:
        self._engine = engine
        self._model = engine._model
        self._device = engine.device
        self._sr = engine.sample_rate
        self.ok = False
        self.error: Optional[str] = None
        self.step_samples = 0
        self.introspection: dict[str, Any] = {}
        self._pending = np.zeros(0, dtype=np.float32)
        self._text = ""
        self._setup()

    def _setup(self) -> None:
        try:
            model = self._model
            enc = model.encoder
            self.introspection = {
                "has_setup_streaming_params": hasattr(enc, "setup_streaming_params"),
                "has_get_initial_cache_state": hasattr(enc, "get_initial_cache_state"),
                "has_conformer_stream_step": hasattr(model, "conformer_stream_step"),
                "has_streaming_cfg": hasattr(enc, "streaming_cfg"),
            }
            if not all(
                (
                    self.introspection["has_setup_streaming_params"],
                    self.introspection["has_get_initial_cache_state"],
                    self.introspection["has_conformer_stream_step"],
                )
            ):
                self.error = "model lacks cache-aware streaming API"
                return

            enc.setup_streaming_params()
            # deterministic per-chunk features (no dithering / extra padding)
            try:
                model.preprocessor.featurizer.dither = 0.0
                model.preprocessor.featurizer.pad_to = 0
            except Exception:  # noqa: BLE001
                pass

            cfg = getattr(enc, "streaming_cfg", None)
            chunk_frames = getattr(cfg, "chunk_size", None) if cfg is not None else None
            if isinstance(chunk_frames, (list, tuple)):
                chunk_frames = chunk_frames[-1]
            sub = int(getattr(enc, "subsampling_factor", 8) or 8)
            window_stride = float(getattr(model.cfg.preprocessor, "window_stride", 0.01))
            hop = max(1, int(self._sr * window_stride))
            # streaming_cfg.chunk_size is in input feature frames (hop-sized),
            # NOT post-subsampling encoder frames, so audio samples per step is
            # chunk_frames * hop (do not multiply by the subsampling factor).
            if chunk_frames:
                self.step_samples = max(hop, int(chunk_frames) * hop)
            else:
                self.step_samples = self._sr  # ~1s fallback
            self.introspection.update(
                {"chunk_frames": chunk_frames, "subsampling": sub, "hop": hop,
                 "step_seconds": round(self.step_samples / self._sr, 3)}
            )

            self._cache_lc, self._cache_lt, self._cache_lc_len = enc.get_initial_cache_state(
                batch_size=1
            )
            self._prev_hyp = None
            self._pred_out = None
            self.ok = True
        except Exception as exc:  # noqa: BLE001
            import traceback

            self.error = f"{type(exc).__name__}: {exc}"
            self.introspection["setup_traceback"] = traceback.format_exc()
            self.ok = False

    def step_sync(self, samples: np.ndarray, is_last: bool) -> str:
        import torch

        model = self._model
        sig = torch.tensor(samples, dtype=torch.float32, device=self._device).unsqueeze(0)
        sig_len = torch.tensor([samples.shape[0]], dtype=torch.int64, device=self._device)
        with torch.inference_mode():
            processed, processed_len = model.preprocessor(input_signal=sig, length=sig_len)
            (
                self._pred_out,
                transcribed,
                self._cache_lc,
                self._cache_lt,
                self._cache_lc_len,
                self._prev_hyp,
            ) = model.conformer_stream_step(
                processed_signal=processed,
                processed_signal_length=processed_len,
                cache_last_channel=self._cache_lc,
                cache_last_time=self._cache_lt,
                cache_last_channel_len=self._cache_lc_len,
                keep_all_outputs=is_last,
                previous_hypotheses=self._prev_hyp,
                previous_pred_out=self._pred_out,
                drop_extra_pre_encoded=None,
                return_transcription=True,
            )
        item = transcribed[0] if isinstance(transcribed, (list, tuple)) else transcribed
        self._text = str(getattr(item, "text", item))
        return self._text

    async def feed(self, audio: np.ndarray) -> None:
        """Buffer incoming audio and decode whole blocks as they complete."""
        if not self.ok:
            return
        self._pending = np.concatenate([self._pending, to_mono_f32(audio)])
        if self._pending.shape[0] < self.step_samples:
            return
        loop = asyncio.get_running_loop()
        async with self._engine._lock:
            while self._pending.shape[0] >= self.step_samples:
                block = self._pending[: self.step_samples]
                self._pending = self._pending[self.step_samples :]
                try:
                    await loop.run_in_executor(
                        self._engine._executor, self.step_sync, block, False
                    )
                except Exception as exc:  # noqa: BLE001
                    self.error = f"step: {type(exc).__name__}: {exc}"
                    self.ok = False
                    return

    async def finalize(self) -> str:
        """Flush the tail and return the final transcript (no full re-decode)."""
        if not self.ok:
            return self._text
        loop = asyncio.get_running_loop()
        block = self._pending if self._pending.shape[0] else np.zeros(1, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        async with self._engine._lock:
            try:
                await loop.run_in_executor(self._engine._executor, self.step_sync, block, True)
            except Exception as exc:  # noqa: BLE001
                self.error = f"final: {type(exc).__name__}: {exc}"
                self.ok = False
        return self._text
