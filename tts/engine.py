"""Qwen3-TTS inference engine.

Wraps `qwen_tts.Qwen3TTSModel` and exposes a small, stable interface used by the
HTTP service:

    engine.synth(text, ...)   -> (np.float32 mono audio, sample_rate)
    engine.stream(text, ...)  -> generator yielding np.float32 mono chunks
    engine.sample_rate        -> int, known after warmup

GPU access is serialised with a lock because a single model instance is not
guaranteed to be thread-safe, and the HTTP service may receive concurrent
requests. If you need true concurrency, run several replicas behind the gateway.
"""

from __future__ import annotations

import os
import threading
from typing import Iterator, Optional

import numpy as np

# Speakers shipped with Qwen3-TTS-12Hz-*-CustomVoice. The two fluent-English
# voices are Ryan and Aiden; we default to one of them so the API always
# produces a consistent English voice regardless of what the client asks for.
SPEAKERS = {
    "Vivian",     # Chinese
    "Serena",     # Chinese
    "Uncle_Fu",   # Chinese
    "Dylan",      # Chinese (Beijing)
    "Eric",       # Chinese (Sichuan)
    "Ryan",       # English  <- default
    "Aiden",      # English
    "Ono_Anna",   # Japanese
    "Sohee",      # Korean
}


def _to_mono_f32(x) -> np.ndarray:
    """Normalise whatever the model returns (tensor / ndarray / list / tuple)
    into a 1-D float32 numpy array in [-1, 1]."""
    # Streaming generators sometimes yield (chunk, sr) tuples.
    if isinstance(x, tuple) and len(x) >= 1:
        x = x[0]
    # torch tensor -> numpy
    if hasattr(x, "detach"):
        x = x.detach().to("cpu").float().numpy()
    arr = np.asarray(x, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim > 1:          # (channels, n) or (n, channels) -> mono
        arr = arr.mean(axis=0 if arr.shape[0] < arr.shape[-1] else -1)
    return np.ascontiguousarray(arr, dtype=np.float32)


class TTSEngine:
    def __init__(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        model_id = os.environ.get(
            "TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        )
        device = os.environ.get("TTS_DEVICE", "cuda:0")
        attn = os.environ.get("TTS_ATTN", "sdpa")  # sdpa | flash_attention_2 | eager
        dtype = getattr(torch, os.environ.get("TTS_DTYPE", "bfloat16"))

        self.default_speaker = os.environ.get("TTS_DEFAULT_SPEAKER", "Ryan")
        self.default_language = os.environ.get("TTS_DEFAULT_LANGUAGE", "English")
        self.default_instruct = os.environ.get("TTS_DEFAULT_INSTRUCT") or None

        self.model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn,
        )
        self._lock = threading.Lock()
        self.sample_rate: Optional[int] = None
        self._optimize(torch)
        self._warmup()

    def _optimize(self, torch) -> None:
        """Speed up autoregressive decode. The bottleneck for a 0.6B model is
        kernel-launch overhead (the GPU starves waiting for the CPU to dispatch
        thousands of tiny ops per step), so torch.compile (CUDA graphs) is the
        big lever. Best-effort: any failure falls back to plain eager mode."""
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception as e:  # noqa: BLE001
            print(f"[tts] tf32 setup skipped: {e}", flush=True)

        # Default OFF: on the custom qwen-tts generate loop, reduce-overhead /
        # CUDA graphs recompile on every changing shape and end up ~2x SLOWER.
        # Opt in with TTS_COMPILE=1 (and ideally TTS_COMPILE_MODE=default).
        if os.environ.get("TTS_COMPILE", "0") != "1":
            print("[tts] torch.compile disabled (TTS_COMPILE!=1)", flush=True)
            return
        mode = os.environ.get("TTS_COMPILE_MODE", "reduce-overhead")
        # Compile the inner generation nn.Module (the hot per-step path). The
        # qwen-tts wrapper holds the real network under one of these attrs.
        for attr in ("model", "lm", "backbone", "transformer", "llm", "net"):
            sub = getattr(self.model, attr, None)
            if isinstance(sub, torch.nn.Module):
                try:
                    setattr(self.model, attr, torch.compile(sub, mode=mode))
                    print(f"[tts] torch.compile applied to .{attr} (mode={mode})", flush=True)
                    return
                except Exception as e:  # noqa: BLE001
                    print(f"[tts] torch.compile on .{attr} failed: {e}", flush=True)
        print("[tts] torch.compile: no compatible submodule found; eager mode", flush=True)

    def _warmup(self) -> None:
        """Run one tiny synthesis so the first real request is fast and so we
        learn the model's output sample rate (needed before streaming)."""
        audio, sr = self.synth("Hello, this is a warmup.")
        self.sample_rate = int(sr)

    def resolve_speaker(self, voice: Optional[str]) -> str:
        """Map an incoming `voice` (which may be an OpenAI voice name like
        'alloy') onto a real Qwen speaker. Unknown -> configured default."""
        if voice and voice in SPEAKERS:
            return voice
        return self.default_speaker

    def synth(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        with self._lock:
            wavs, sr = self.model.generate_custom_voice(
                text=text,
                language=language or self.default_language,
                speaker=speaker or self.default_speaker,
                instruct=instruct if instruct is not None else self.default_instruct,
            )
        audio = _to_mono_f32(wavs[0] if isinstance(wavs, (list, tuple)) else wavs)
        return audio, int(sr)

    def stream(
        self,
        text: str,
        language: Optional[str] = None,
        speaker: Optional[str] = None,
        instruct: Optional[str] = None,
    ) -> Iterator[np.ndarray]:
        """Yield audio chunks as they are produced.

        If the installed `qwen-tts` supports `generate_custom_voice(..., stream=True)`
        we use the native low-latency path. Otherwise we transparently fall back
        to full synthesis sliced into fixed-size frames, so the HTTP contract is
        identical either way.
        """
        kwargs = dict(
            text=text,
            language=language or self.default_language,
            speaker=speaker or self.default_speaker,
            instruct=instruct if instruct is not None else self.default_instruct,
        )

        with self._lock:
            try:
                gen = self.model.generate_custom_voice(stream=True, **kwargs)
                produced = False
                for chunk in gen:
                    produced = True
                    arr = _to_mono_f32(chunk)
                    if arr.size:
                        yield arr
                if produced:
                    return
            except TypeError:
                # `stream=True` not supported by this qwen-tts version.
                pass

            # Fallback: synthesise fully, then emit in frames.
            wavs, sr = self.model.generate_custom_voice(**kwargs)

        audio = _to_mono_f32(wavs[0] if isinstance(wavs, (list, tuple)) else wavs)
        frame = max(1, int(int(sr) * float(os.environ.get("TTS_STREAM_FRAME_SEC", "0.2"))))
        for i in range(0, len(audio), frame):
            yield audio[i : i + frame]
