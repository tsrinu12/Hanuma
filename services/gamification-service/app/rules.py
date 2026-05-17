"""Points + streaks rules engine.

All state lives in Postgres; Redis is only used for short-window dedupe so
hot-view spikes don't all roundtrip to the DB.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Dict, List, Optional, Tuple

import redis
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import PointEvent, UserBadge, UserPoints, UserStreak, VideoStats, WatchSession

log = logging.getLogger("gamification.rules")
_r = redis.Redis.from_url(settings.redis_url)


def _dedupe_key(user_id: str, video_id: str) -> str:
    return f"gam:viewdedupe:{user_id}:{video_id}"


def _level_for_points(points: int) -> int:
    """Simple log-ish curve: every 100 points = 1 level, capped at 50."""
    return min(50, 1 + points // 100)


# ---- public mutators -----------------------------------------------------


def record_watch_event(
    db: Session, user_id: str, video_id: str, fraction_watched: float, watch_seconds: int = 0
) -> Dict:
    """Called from a `/event` watch heartbeat. Decides whether to count a view
    and/or complete and awards points accordingly. Idempotent within
    `dedupe_window` for views."""

    awards: List[Tuple[str, int]] = []
    counted_view = 0
    counted_complete = 0

    if fraction_watched >= settings.min_view_fraction:
        # Dedupe so reloading the page doesn't farm views
        first_view = _r.set(_dedupe_key(user_id, video_id), "1", ex=settings.dedupe_window, nx=True)
        if first_view:
            counted_view = 1
            awards.append(("view", settings.points_view))
            _bump_video(db, video_id, views=1, unique_viewers=1)

            # First view of the day = bonus
            today = dt.date.today()
            first_day_key = f"gam:firstviewday:{user_id}:{today.isoformat()}"
            if _r.set(first_day_key, "1", ex=86400, nx=True):
                awards.append(("first_view_of_day", settings.points_first_view_of_day))
                streak_bonus = _bump_streak(db, user_id, today)
                if streak_bonus > 0:
                    awards.append(("streak_bonus", streak_bonus))
        else:
            _bump_video(db, video_id, views=1)  # raw view counter still increments

    if fraction_watched >= settings.min_complete_fraction:
        counted_complete = 1
        awards.append(("complete", settings.points_complete))
        _bump_video(db, video_id, completes=1, watch_seconds=watch_seconds)
    elif watch_seconds:
        _bump_video(db, video_id, watch_seconds=watch_seconds)

    db.add(WatchSession(
        user_id=user_id, video_id=video_id,
        fraction_watched=int(min(100, max(0, fraction_watched * 100))),
        counted_as_view=counted_view, counted_as_complete=counted_complete,
    ))

    total_points = _award_many(db, user_id, video_id, awards)
    db.commit()
    return {
        "user_id": user_id, "video_id": video_id,
        "counted_view": counted_view, "counted_complete": counted_complete,
        "awards": [{"action": a, "points": p} for a, p in awards],
        "total_points": total_points,
    }


def award_action(db: Session, user_id: str, action: str, video_id: Optional[str]) -> Dict:
    pts_map = {
        "like": settings.points_like, "comment": settings.points_comment,
        "share": settings.points_share, "upload": settings.points_upload,
    }
    if action not in pts_map:
        return {"error": f"unknown action {action}"}
    if video_id:
        kwargs = {f"{action}s": 1} if action in {"like", "comment", "share"} else {}
        if kwargs:
            _bump_video(db, video_id, **kwargs)
    total = _award_many(db, user_id, video_id, [(action, pts_map[action])])
    db.commit()
    return {"user_id": user_id, "action": action, "points": pts_map[action], "total_points": total}


# ---- helpers -------------------------------------------------------------


def _bump_video(db: Session, video_id: str, **deltas: int) -> None:
    stats = db.get(VideoStats, video_id)
    if stats is None:
        stats = VideoStats(video_id=video_id)
        db.add(stats)
        db.flush()
    for k, v in deltas.items():
        setattr(stats, k, (getattr(stats, k) or 0) + v)


def _bump_streak(db: Session, user_id: str, today: dt.date) -> int:
    s = db.get(UserStreak, user_id)
    if s is None:
        s = UserStreak(user_id=user_id, current_streak=1, longest_streak=1, last_active_day=today)
        db.add(s)
        return 0
    if s.last_active_day == today:
        return 0
    if s.last_active_day == today - dt.timedelta(days=1):
        s.current_streak += 1
    else:
        s.current_streak = 1
    s.longest_streak = max(s.longest_streak, s.current_streak)
    s.last_active_day = today
    # Bonus scales with streak length (capped)
    return min(s.current_streak, 30) * settings.points_streak_bonus_per_day


def _award_many(db: Session, user_id: str, video_id: Optional[str], awards: List[Tuple[str, int]]) -> int:
    if not awards:
        up = db.get(UserPoints, user_id)
        return up.points if up else 0
    for action, pts in awards:
        db.add(PointEvent(user_id=user_id, video_id=video_id, action=action, points=pts))
    up = db.get(UserPoints, user_id)
    if up is None:
        up = UserPoints(user_id=user_id, points=0)
        db.add(up)
        db.flush()
    up.points = (up.points or 0) + sum(p for _, p in awards)
    up.level = _level_for_points(up.points)
    return up.points
