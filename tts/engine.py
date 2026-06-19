"""Kokoro-82M TTS engine.

Small, fast, high-quality English TTS. Exposes the same interface the HTTP
service expects:

    engine.synth(text, ...)   -> (np.float32 mono audio, sample_rate)
    engine.stream(text, ...)  -> generator yielding np.float32 mono chunks
    engine.sample_rate        -> 24000

Kokoro's pipeline already yields audio per sentence/segment, so streaming is
native (low latency to first chunk). GPU access is serialised with a lock.
"""

from __future__ import annotations

import os
import threading
from typing import Iterator, Optional

import numpy as np

# Kokoro voices. Prefix: a=American / b=British English; f=female / m=male.
# (Other languages exist via other lang_codes; we ship the English set.)
SPEAKERS = {
    # American female
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "af_aoede",
    # American male
    "am_michael", "am_adam", "am_eric", "am_liam", "am_onyx", "am_fenrir", "am_puck",
    # British female / male
    "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
}
ENGLISH_VOICES = ["am_michael", "am_adam", "bm_george", "af_heart", "bf_emma"]


def _to_mono_f32(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().to("cpu").float().numpy()
    arr = np.asarray(x, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return np.ascontiguousarray(arr, dtype=np.float32)


class TTSEngine:
    SAMPLE_RATE = 24000  # Kokoro outputs 24 kHz

    def __init__(self) -> None:
        from kokoro import KPipeline

        # lang_code 'a' = American English, 'b' = British English.
        self.lang_code = os.environ.get("TTS_LANG_CODE", "a")
        device = os.environ.get("TTS_DEVICE", "cuda") or None
        self.default_speaker = os.environ.get("TTS_DEFAULT_SPEAKER", "am_michael")
        self.default_language = os.environ.get("TTS_DEFAULT_LANGUAGE", "English")
        self.default_speed = float(os.environ.get("TTS_SPEED", "1.0"))

        self.pipeline = KPipeline(lang_code=self.lang_code, device=device)
        self._lock = threading.Lock()
        self.sample_rate = self.SAMPLE_RATE
        self._warmup()

    def _warmup(self) -> None:
        try:
            self.synth("Hello, this is a warmup.")
        except Exception as e:  # noqa: BLE001
            print(f"[tts] warmup skipped: {e}", flush=True)

    def resolve_speaker(self, voice: Optional[str]) -> str:
        """Map an incoming voice onto a real Kokoro voice. Unknown names (e.g.
        OpenAI 'alloy', or the old Qwen 'Ryan') fall back to the default, so a
        single consistent English voice is guaranteed."""
        if voice and voice in SPEAKERS:
            return voice
        return self.default_speaker

    def _generate(self, text, voice, speed) -> Iterator[np.ndarray]:
        for result in self.pipeline(text, voice=voice, speed=speed):
            # Newer Kokoro yields a Result object (.audio); older yields a
            # (graphemes, phonemes, audio) tuple.
            audio = getattr(result, "audio", None)
            if audio is None:
                audio = result[-1] if isinstance(result, (tuple, list)) else result
            arr = _to_mono_f32(audio)
            if arr.size:
                yield arr

    def synth(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,  # unused (Kokoro has no instruct)
    ) -> tuple[np.ndarray, int]:
        voice = self.resolve_speaker(speaker)
        with self._lock:
            segments = list(self._generate(text, voice, self.default_speed))
        audio = (
            np.concatenate(segments) if segments else np.zeros(1, dtype=np.float32)
        )
        return audio, self.SAMPLE_RATE

    def stream(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> Iterator[np.ndarray]:
        """Native streaming: Kokoro emits audio per segment as it is produced."""
        voice = self.resolve_speaker(speaker)
        with self._lock:
            for chunk in self._generate(text, voice, self.default_speed):
                yield chunk
