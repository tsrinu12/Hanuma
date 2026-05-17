import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from .db import Base


def _uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    keycloak_sub = Column(String, unique=True, index=True, nullable=True)
    email = Column(String, unique=True, index=True)
    display_name = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class Video(Base):
    __tablename__ = "videos"

    id = Column(String, primary_key=True, default=_uuid)
    owner_id = Column(String, ForeignKey("users.id"), index=True, nullable=True)
    title = Column(String, nullable=False)
    description = Column(Text)
    status = Column(String, default="uploading")  # uploading|processing|ready|failed|blocked
    duration_seconds = Column(Float)
    raw_s3_key = Column(String)
    hls_master_url = Column(String)
    thumbnail_url = Column(String)
    tags = Column(JSONB, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(String, primary_key=True, default=_uuid)
    video_id = Column(String, ForeignKey("videos.id"), index=True)
    language = Column(String, default="en")
    full_text = Column(Text)
    segments = Column(JSONB)  # [{start, end, text}]
    created_at = Column(DateTime, default=datetime.utcnow)


class AIOutput(Base):
    __tablename__ = "ai_outputs"

    id = Column(String, primary_key=True, default=_uuid)
    video_id = Column(String, ForeignKey("videos.id"), index=True)
    kind = Column(String)  # summary | moderation | highlights
    payload = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)


class WatchEvent(Base):
    __tablename__ = "watch_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    video_id = Column(String, ForeignKey("videos.id"), index=True)
    watched_seconds = Column(Float, default=0.0)
    completed = Column(Integer, default=0)  # 0/1
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
