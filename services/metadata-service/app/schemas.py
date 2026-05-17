from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict


class VideoCreate(BaseModel):
    title: str
    description: Optional[str] = None
    owner_id: Optional[str] = None
    raw_s3_key: Optional[str] = None
    tags: List[str] = []


class VideoUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    duration_seconds: Optional[float] = None
    hls_master_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    tags: Optional[List[str]] = None


class VideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    description: Optional[str]
    status: str
    duration_seconds: Optional[float]
    hls_master_url: Optional[str]
    thumbnail_url: Optional[str]
    tags: Optional[List[str]] = []
    created_at: datetime


class TranscriptIn(BaseModel):
    video_id: str
    language: str = "en"
    full_text: str
    segments: List[dict] = []


class AIOutputIn(BaseModel):
    video_id: str
    kind: str
    payload: Any


class WatchEventIn(BaseModel):
    user_id: str
    video_id: str
    watched_seconds: float = 0.0
    completed: bool = False
