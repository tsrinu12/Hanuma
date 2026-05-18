"""Schemas for the live moderation service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


EventType = Literal["TRANSCRIPT", "FLAG", "STATUS"]
Action = Literal["NONE", "WARN", "BLOCK"]


class TranscriptEvent(BaseModel):
    event: Literal["TRANSCRIPT"] = "TRANSCRIPT"
    chunk_id: int
    start_s: float
    end_s: float
    text: str


class FlagEvent(BaseModel):
    event: Literal["FLAG"] = "FLAG"
    chunk_id: int
    start_s: float
    end_s: float
    severity: float = Field(ge=0.0, le=1.0)
    category: str
    action: Action
    explanation: str


class StatusEvent(BaseModel):
    event: Literal["STATUS"] = "STATUS"
    msg: str


class StreamControl(BaseModel):
    """First text frame on the websocket — declares stream params."""

    stream_id: str
    sample_rate: int = 16000
