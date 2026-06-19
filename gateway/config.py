"""Runtime configuration for the voice API."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    api_key: str = ""
    gateway_timeout: float = 600.0

    llm_base_url: str = "http://llm:8080"
    llm_model_name: str = "gemma-3-12b"

    tts_model_name: str = "kokoro"
    tts_default_speaker: str = "am_michael"
    tts_default_language: str = "English"
    tts_device: str = "cuda"

    stt_model_name: str = "nemotron-speech-streaming-en-0.6b"
    stt_model_id: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    stt_device: str = "cuda"
    stt_sample_rate: int = 16000

    vad_model_name: str = "silero-vad-onnx"
    vad_sample_rate: int = 16000
    vad_confidence: float = 0.7
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.2
    vad_min_volume: float = 0.6

    turn_model_name: str = "smart-turn-v3"
    turn_cpu_count: int = Field(default=1, ge=1)
    turn_stop_secs: float = 3.0
    turn_pre_speech_ms: float = 500.0
    turn_max_duration_secs: float = 8.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
