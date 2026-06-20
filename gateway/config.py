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

    llm_base_url: str = "http://127.0.0.1:8081"
    llm_model_name: str = "gemma-4-12b"

    tts_model_name: str = "kokoro"
    tts_default_speaker: str = "am_michael"
    tts_default_language: str = "English"
    tts_device: str = "cuda"

    stt_model_name: str = "parakeet-realtime-eou-120m"
    stt_model_id: str = "nvidia/parakeet_realtime_eou_120m-v1"
    stt_device: str = "cuda"
    stt_sample_rate: int = 16000
    # Parakeet EOU emits these inline markers in the decoded text; they are
    # stripped from partial/final text and surfaced as endpoint signals.
    # `</s>` behaves like an end-of-utterance marker on this vocab.
    stt_eou_tokens: tuple[str, ...] = ("<EOU>", "</s>")
    stt_eob_token: str = "<EOB>"

    vad_model_name: str = "silero-vad-onnx"
    vad_sample_rate: int = 16000
    vad_confidence: float = 0.7
    vad_start_secs: float = 0.2
    vad_stop_secs: float = 0.2
    vad_min_volume: float = 0.6

    turn_model_name: str = "smart-turn-v3.2"
    turn_cpu_count: int = Field(default=1, ge=1)
    turn_stop_secs: float = 3.0
    turn_pre_speech_ms: float = 500.0
    turn_max_duration_secs: float = 8.0

    # Pause-tolerant endpoint controller: a Parakeet <EOU> is a *candidate* for
    # end-of-thought; we only fire `thought_end` once Smart Turn confirms it (or
    # the max-wait ceiling elapses so a stubborn "incomplete" can never hang).
    endpoint_require_smart_turn: bool = True
    endpoint_max_wait_secs: float = 4.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
