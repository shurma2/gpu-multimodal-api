"""Audio decoding and conversion helpers."""

from __future__ import annotations

import io
import subprocess
from typing import Optional

import numpy as np
import soundfile as sf


class AudioDecodeError(RuntimeError):
    """Raised when uploaded audio cannot be decoded."""


def to_mono_f32(audio: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    elif arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
    return np.ascontiguousarray(arr, dtype=np.float32)


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if source_sr == target_sr:
        return np.ascontiguousarray(audio, dtype=np.float32)
    try:
        import soxr

        return np.ascontiguousarray(
            soxr.resample(audio, source_sr, target_sr, quality="HQ"),
            dtype=np.float32,
        )
    except Exception:
        duration = len(audio) / float(source_sr)
        target_len = max(1, int(round(duration * target_sr)))
        src_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        dst_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
        return np.ascontiguousarray(np.interp(dst_x, src_x, audio).astype(np.float32))


def decode_audio_bytes(data: bytes, target_sr: Optional[int] = 16000) -> tuple[np.ndarray, int]:
    """Decode common audio containers into mono float32 PCM."""
    try:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        audio = to_mono_f32(audio)
    except Exception as sf_error:
        if target_sr is None:
            target_sr = 16000
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    "pipe:0",
                    "-f",
                    "f32le",
                    "-ac",
                    "1",
                    "-ar",
                    str(target_sr),
                    "pipe:1",
                ],
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            audio = np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32)
            sr = target_sr
        except Exception as ffmpeg_error:
            raise AudioDecodeError(
                f"could not decode audio with soundfile or ffmpeg: {sf_error}; {ffmpeg_error}"
            ) from ffmpeg_error

    if target_sr is not None and sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr

    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def float_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(to_mono_f32(audio), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def pcm16_bytes_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
