"""CPU VAD and end-of-turn helpers backed by Pipecat ONNX models."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from gateway.config import Settings
from voice.audio import decode_audio_bytes, float_to_pcm16_bytes


class VADTurnService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._executor = ThreadPoolExecutor(max_workers=max(1, settings.turn_cpu_count))

    def create_vad(self, sample_rate: int = 16000):
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams

        analyzer = SileroVADAnalyzer(
            sample_rate=sample_rate,
            params=VADParams(
                confidence=self.settings.vad_confidence,
                start_secs=self.settings.vad_start_secs,
                stop_secs=self.settings.vad_stop_secs,
                min_volume=self.settings.vad_min_volume,
            ),
        )
        analyzer.set_sample_rate(sample_rate)
        return analyzer

    def create_turn_analyzer(self, sample_rate: int = 16000):
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        return LocalSmartTurnAnalyzerV3(
            sample_rate=sample_rate,
            cpu_count=self.settings.turn_cpu_count,
            params=SmartTurnParams(
                stop_secs=self.settings.turn_stop_secs,
                pre_speech_ms=self.settings.turn_pre_speech_ms,
                max_duration_secs=self.settings.turn_max_duration_secs,
            ),
        )

    async def analyze_vad_bytes(self, data: bytes) -> dict[str, Any]:
        from pipecat.audio.vad.vad_analyzer import VADState

        audio, sr = decode_audio_bytes(data, target_sr=self.settings.vad_sample_rate)
        pcm = float_to_pcm16_bytes(audio)
        analyzer = self.create_vad(sr)
        frame_bytes = analyzer.num_frames_required() * 2

        states: list[str] = []
        speech_frames = 0
        total_frames = 0
        for offset in range(0, len(pcm), frame_bytes):
            chunk = pcm[offset : offset + frame_bytes]
            if len(chunk) < frame_bytes:
                chunk = chunk + b"\x00" * (frame_bytes - len(chunk))
            state = await analyzer.analyze_audio(chunk)
            states.append(state.name.lower())
            total_frames += 1
            if state in (VADState.STARTING, VADState.SPEAKING, VADState.STOPPING):
                speech_frames += 1

        return {
            "model": self.settings.vad_model_name,
            "sample_rate": sr,
            "speech_detected": speech_frames > 0,
            "speech_ratio": speech_frames / total_frames if total_frames else 0.0,
            "final_state": states[-1] if states else "quiet",
            "states": states,
        }

    async def predict_turn_bytes(self, data: bytes) -> dict[str, Any]:
        audio, sr = decode_audio_bytes(data, target_sr=self.settings.vad_sample_rate)
        analyzer = self.create_turn_analyzer(sr)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(self._executor, analyzer._predict_endpoint, audio)
        return {
            "model": self.settings.turn_model_name,
            "sample_rate": sr,
            "complete": bool(result["prediction"] == 1),
            "probability": float(result["probability"]),
        }
