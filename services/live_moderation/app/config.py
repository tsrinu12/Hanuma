"""Live moderation service settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field

from shared.config import BaseServiceSettings


class Settings(BaseServiceSettings):
    service_name: str = Field(default="live_moderation")
    port: int = Field(default=8004)

    # --- ASR ---
    asr_model_lite: str = Field(default="tiny.en")
    asr_model_prod: str = Field(default="medium.en")
    asr_sample_rate: int = Field(default=16000)

    # --- VAD chunking ---
    # Target chunk size in seconds. We finalize a chunk when we hit either
    # ``chunk_target_s`` of voiced audio or ``silence_break_s`` of silence
    # after at least ``min_voiced_s`` of speech.
    chunk_target_s: float = Field(default=2.5)
    silence_break_s: float = Field(default=0.4)
    min_voiced_s: float = Field(default=0.5)

    # Energy threshold (RMS) below which a frame is considered silence.
    # VAD in lite mode is energy-based; prod uses Silero or WebRTC VAD.
    vad_energy_threshold: float = Field(default=0.005)

    # --- Moderation thresholds ---
    block_above: float = Field(default=0.85)
    warn_above: float = Field(default=0.5)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
