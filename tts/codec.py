"""Audio encoders for TTS responses."""

from __future__ import annotations

import io
import subprocess
import threading
from typing import Iterator

import numpy as np
import soundfile as sf

FORMATS = {
    "wav": (None, "audio/wav"),
    "pcm": (None, "audio/L16"),
    "flac": (["-f", "flac"], "audio/flac"),
    "mp3": (["-f", "mp3", "-b:a", "128k"], "audio/mpeg"),
    "opus": (["-f", "ogg", "-c:a", "libopus", "-b:a", "64k"], "audio/ogg"),
    "aac": (["-f", "adts", "-c:a", "aac", "-b:a", "128k"], "audio/aac"),
}


def to_s16le(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def encode_full(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    if fmt == "pcm":
        return to_s16le(audio)
    if fmt in ("wav", "flac"):
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format=fmt.upper(), subtype="PCM_16" if fmt == "wav" else None)
        return buf.getvalue()
    out_args, _ = FORMATS[fmt]
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            *out_args,
            "pipe:1",
        ],
        input=to_s16le(audio),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def wav_header(sr: int, channels: int = 1, bits: int = 16) -> bytes:
    import struct

    byte_rate = sr * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sr, byte_rate, block_align, bits)
        + b"data"
        + struct.pack("<I", 0xFFFFFFFF)
    )


def stream_native(chunks: Iterator[np.ndarray], sr: int, fmt: str) -> Iterator[bytes]:
    if fmt == "wav":
        yield wav_header(sr)
    for chunk in chunks:
        if chunk.size:
            yield to_s16le(chunk)


def stream_ffmpeg(chunks: Iterator[np.ndarray], sr: int, fmt: str) -> Iterator[bytes]:
    out_args, _ = FORMATS[fmt]
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sr),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            *out_args,
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    def feed() -> None:
        try:
            for chunk in chunks:
                if chunk.size:
                    proc.stdin.write(to_s16le(chunk))
        except Exception:
            pass
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    threading.Thread(target=feed, daemon=True).start()
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
