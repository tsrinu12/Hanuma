"""Moderation service settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field

from shared.config import BaseServiceSettings


class Settings(BaseServiceSettings):
    service_name: str = Field(default="moderation")
    port: int = Field(default=8003)

    # --- Decision thresholds. Per-category, applied AFTER fusion. ---
    # severity in [0.0, 1.0]. Below ``allow_below`` -> ALLOW.
    # Between ``allow_below`` and ``block_above`` -> REVIEW.
    # Above ``block_above`` -> BLOCK.
    allow_below: float = Field(default=0.35)
    block_above: float = Field(default=0.85)

    # --- Per-modality weights in the fusion layer. ---
    # We use a soft-OR fusion: each modality contributes its own severity
    # scaled by its weight, then we combine via max-of-weighted plus a
    # complementary noisy-OR term for cross-modal agreement. See fusion.py.
    weight_text: float = Field(default=1.0)
    weight_visual: float = Field(default=1.0)
    weight_audio: float = Field(default=0.8)

    # --- Models ---
    text_model_lite: str = Field(default="unitary/toxic-bert")
    text_model_prod: str = Field(default="unitary/toxic-bert")
    visual_model_prod: str = Field(default="openai/clip-vit-base-patch32")
    # Audio: in prod we'd use PANNs CNN14. Lite uses a hand-coded mel-stat rule.

    # --- Sampling ---
    # In lite mode we run the visual classifier over keyframe embeddings
    # supplied by the caller. The caller is expected to sample one frame
    # every N seconds during ingest. We record the configured cadence here
    # for explanation construction.
    keyframe_period_s: float = Field(default=2.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
