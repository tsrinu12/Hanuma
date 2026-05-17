"""Metadata service - source of truth for videos, users, transcripts, AI outputs."""
import logging
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .db import Base, engine, get_db
from .models import AIOutput, Transcript, Video, WatchEvent
from .schemas import (AIOutputIn, TranscriptIn, VideoCreate, VideoOut,
                      VideoUpdate, WatchEventIn)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("metadata")

app = FastAPI(title="distrebute.com - Metadata Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=False,
)


@app.on_event("startup")
def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    log.info("Metadata service started; schema ensured.")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# ---------- Videos ----------
@app.post("/videos", response_model=VideoOut, status_code=201)
def create_video(payload: VideoCreate, db: Session = Depends(get_db)):
    v = Video(**payload.model_dump())
    db.add(v); db.commit(); db.refresh(v)
    return v


@app.get("/videos", response_model=List[VideoOut])
def list_videos(
    status: str | None = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(Video).order_by(Video.created_at.desc())
    if status:
        q = q.filter(Video.status == status)
    return q.offset(offset).limit(limit).all()


@app.get("/videos/{video_id}", response_model=VideoOut)
def get_video(video_id: str, db: Session = Depends(get_db)):
    v = db.query(Video).filter(Video.id == video_id).first()
    if not v:
        raise HTTPException(404, "video not found")
    return v


@app.patch("/videos/{video_id}", response_model=VideoOut)
def update_video(video_id: str, payload: VideoUpdate, db: Session = Depends(get_db)):
    v = db.query(Video).filter(Video.id == video_id).first()
    if not v:
        raise HTTPException(404, "video not found")
    for k, val in payload.model_dump(exclude_unset=True).items():
        setattr(v, k, val)
    db.commit(); db.refresh(v)
    return v


# ---------- Transcripts ----------
@app.post("/transcripts", status_code=201)
def upsert_transcript(payload: TranscriptIn, db: Session = Depends(get_db)):
    t = Transcript(**payload.model_dump())
    db.add(t); db.commit(); db.refresh(t)
    return {"id": t.id}


@app.get("/transcripts/{video_id}")
def get_transcript(video_id: str, db: Session = Depends(get_db)):
    t = (
        db.query(Transcript)
        .filter(Transcript.video_id == video_id)
        .order_by(Transcript.created_at.desc())
        .first()
    )
    if not t:
        raise HTTPException(404, "transcript not found")
    return {
        "id": t.id, "video_id": t.video_id, "language": t.language,
        "full_text": t.full_text, "segments": t.segments,
    }


# ---------- AI outputs (summary, moderation, highlights) ----------
@app.post("/ai-outputs", status_code=201)
def add_ai_output(payload: AIOutputIn, db: Session = Depends(get_db)):
    a = AIOutput(**payload.model_dump())
    db.add(a); db.commit(); db.refresh(a)
    return {"id": a.id}


@app.get("/ai-outputs/{video_id}")
def list_ai_outputs(video_id: str, db: Session = Depends(get_db)):
    rows = db.query(AIOutput).filter(AIOutput.video_id == video_id).all()
    return [{"kind": r.kind, "payload": r.payload, "created_at": r.created_at} for r in rows]


# ---------- Watch events (used by recommendations) ----------
@app.post("/watch-events", status_code=201)
def log_watch_event(payload: WatchEventIn, db: Session = Depends(get_db)):
    e = WatchEvent(
        user_id=payload.user_id, video_id=payload.video_id,
        watched_seconds=payload.watched_seconds, completed=int(payload.completed),
    )
    db.add(e); db.commit()
    return {"ok": True}


@app.get("/watch-events")
def list_watch_events(user_id: str | None = None, limit: int = 1000, db: Session = Depends(get_db)):
    q = db.query(WatchEvent)
    if user_id:
        q = q.filter(WatchEvent.user_id == user_id)
    rows = q.order_by(WatchEvent.created_at.desc()).limit(limit).all()
    return [
        {"user_id": r.user_id, "video_id": r.video_id,
         "watched_seconds": r.watched_seconds, "completed": bool(r.completed)}
        for r in rows
    ]
