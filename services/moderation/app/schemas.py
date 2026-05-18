"""Schemas for the moderation service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Modality = Literal["text", "visual", "audio"]
Category = Literal[
    "toxicity",
    "harassment",
    "violence",
    "sexual",
    "self_harm",
    "hate",
    "spam",
]
Decision = Literal["ALLOW", "REVIEW", "BLOCK"]


# ---------- inputs ----------


class TranscriptSegment(BaseModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    text: str


class Keyframe(BaseModel):
    """A sampled frame from the video.

    The caller may supply either a precomputed CLIP embedding *or* a base64
    PNG (only the latter requires the visual model — lite mode only handles
    embeddings).
    """

    t_s: float
    embedding: list[float] | None = None
    image_b64: str | None = None


class AudioSegment(BaseModel):
    """A short audio window.

    Lite mode classifies from features supplied by the caller. Prod loads
    PANNs and decodes from the raw waveform.
    """

    start_s: float
    end_s: float
    # mel statistics: [mean_db, std_db, peak_db, percussive_ratio, voicing_ratio]
    # Useful only in lite mode; prod ignores and uses ``wave_b64``.
    features: list[float] | None = None
    wave_b64: str | None = None


class ModerationRequest(BaseModel):
    video_id: str
    duration_s: float
    transcript: list[TranscriptSegment] = []
    keyframes: list[Keyframe] = []
    audio_segments: list[AudioSegment] = []


# ---------- outputs ----------


class ModalityFlag(BaseModel):
    """One reason cited by one modality."""

    modality: Modality
    category: Category
    severity: float                  # in [0, 1]
    start_s: float
    end_s: float
    # Short, human-readable explanation for the creator UI.
    explanation: str


class ModerationResponse(BaseModel):
    video_id: str
    decision: Decision
    severity: float                  # fused, in [0, 1]
    # The flags driving the decision, sorted by severity desc.
    flags: list[ModalityFlag]
    # Per-modality top-line severity (useful for an "x-ray" creator view).
    per_modality: dict[str, float]


class HealthDetails(BaseModel):
    text_loaded: bool
    visual_loaded: bool
    audio_loaded: bool
