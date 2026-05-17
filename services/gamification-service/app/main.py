"""gamification-service: viewer counts, points, streaks, leaderboard.

Endpoints:
  POST /event              body: {user_id, video_id, fraction_watched, watch_seconds}
  POST /action             body: {user_id, action, video_id?}    (like / comment / share / upload)
  GET  /stats/{video_id}                                          -> per-video aggregates
  GET  /points/{user_id}                                          -> total points + level + streak
  GET  /leaderboard?limit=50                                      -> top users
  GET  /badges/{user_id}                                          -> earned badges
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import (
    Badge,
    SessionLocal,
    UserBadge,
    UserPoints,
    UserStreak,
    VideoStats,
    init_db,
)
from .rules import award_action, record_watch_event

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gamification-service")

app = FastAPI(title="distrebute.com - Gamification Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup() -> None:
    init_db()


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---- schemas -------------------------------------------------------------


class WatchEvent(BaseModel):
    user_id: str
    video_id: str
    fraction_watched: float
    watch_seconds: int = 0


class ActionEvent(BaseModel):
    user_id: str
    action: str  # like | comment | share | upload
    video_id: Optional[str] = None


# ---- routes --------------------------------------------------------------


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/event")
def event(req: WatchEvent, db: Session = Depends(get_db)):
    return record_watch_event(
        db, req.user_id, req.video_id, req.fraction_watched, req.watch_seconds
    )


@app.post("/action")
def action(req: ActionEvent, db: Session = Depends(get_db)):
    out = award_action(db, req.user_id, req.action, req.video_id)
    if "error" in out:
        raise HTTPException(400, out["error"])
    return out


@app.get("/stats/{video_id}")
def stats(video_id: str, db: Session = Depends(get_db)):
    s = db.get(VideoStats, video_id)
    if s is None:
        return {"video_id": video_id, "views": 0, "unique_viewers": 0,
                "completes": 0, "likes": 0, "comments": 0, "shares": 0, "watch_seconds": 0}
    return {
        "video_id": s.video_id, "views": s.views, "unique_viewers": s.unique_viewers,
        "completes": s.completes, "likes": s.likes, "comments": s.comments,
        "shares": s.shares, "watch_seconds": s.watch_seconds,
    }


@app.get("/points/{user_id}")
def points(user_id: str, db: Session = Depends(get_db)):
    up = db.get(UserPoints, user_id)
    streak = db.get(UserStreak, user_id)
    return {
        "user_id": user_id,
        "points": up.points if up else 0,
        "level": up.level if up else 1,
        "current_streak": streak.current_streak if streak else 0,
        "longest_streak": streak.longest_streak if streak else 0,
    }


@app.get("/leaderboard")
def leaderboard(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.execute(
        select(UserPoints).order_by(UserPoints.points.desc()).limit(limit)
    ).scalars().all()
    return {
        "top": [
            {"rank": i + 1, "user_id": r.user_id, "points": r.points, "level": r.level}
            for i, r in enumerate(rows)
        ]
    }


@app.get("/badges/{user_id}")
def badges(user_id: str, db: Session = Depends(get_db)):
    rows = (
        db.execute(
            select(UserBadge, Badge)
            .join(Badge, Badge.code == UserBadge.badge_code)
            .where(UserBadge.user_id == user_id)
        )
        .all()
    )
    return {
        "user_id": user_id,
        "badges": [
            {"code": b.code, "name": b.name, "description": b.description, "awarded_at": ub.awarded_at}
            for ub, b in rows
        ],
    }
