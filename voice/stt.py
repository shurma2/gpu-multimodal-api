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
