"""Rule-based spam heuristics applied alongside the ML models.

These catch the obvious patterns (URL stuffing, caps lock screaming, emoji
spam, repeated-character padding, posting rate) cheaply, before falling
through to the more expensive transformer classifiers.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List

import redis

from .config import settings

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "☀-⛿✀-➿"
    "]+",
    flags=re.UNICODE,
)
REPEAT_RE = re.compile(r"(.)\1{%d,}" % (settings.max_repeat_char_run - 1))

_r = redis.Redis.from_url(settings.redis_url)


def features(text: str) -> Dict[str, float]:
    text = text or ""
    length = max(len(text), 1)
    urls = URL_RE.findall(text)
    emojis = EMOJI_RE.findall(text)
    emoji_chars = sum(len(e) for e in emojis)
    letters = [c for c in text if c.isalpha()]
    caps = sum(1 for c in letters if c.isupper())
    caps_ratio = caps / max(len(letters), 1)
    emoji_ratio = emoji_chars / length
    has_repeat_run = bool(REPEAT_RE.search(text))
    return {
        "length": length,
        "url_count": len(urls),
        "caps_ratio": caps_ratio,
        "emoji_ratio": emoji_ratio,
        "has_repeat_run": int(has_repeat_run),
    }


def rule_scores(text: str) -> Dict[str, float]:
    f = features(text)
    flags: Dict[str, float] = {}
    if f["url_count"] > settings.max_urls_per_comment:
        flags["too_many_urls"] = float(f["url_count"])
    if f["caps_ratio"] > settings.max_caps_ratio and f["length"] > 12:
        flags["all_caps"] = f["caps_ratio"]
    if f["emoji_ratio"] > settings.max_emoji_ratio and f["length"] > 12:
        flags["emoji_flood"] = f["emoji_ratio"]
    if f["has_repeat_run"]:
        flags["repeat_run"] = 1.0
    return flags


def rate_limited(user_id: str) -> bool:
    """Sliding-window rate limit in Redis."""
    if not user_id:
        return False
    now = int(time.time())
    key = f"spam:rate:{user_id}"
    pipe = _r.pipeline()
    pipe.zremrangebyscore(key, 0, now - settings.rate_window_seconds)
    pipe.zadd(key, {str(now): now})
    pipe.zcard(key)
    pipe.expire(key, settings.rate_window_seconds * 2)
    _, _, count, _ = pipe.execute()
    return int(count) > settings.rate_max_in_window


def repeat_post(user_id: str, text: str) -> bool:
    """Detect a user posting the same content twice in a short window."""
    if not user_id or not text:
        return False
    key = f"spam:lastpost:{user_id}"
    prev = _r.get(key)
    _r.setex(key, settings.rate_window_seconds, text[:512])
    return prev is not None and prev.decode("utf-8", "ignore") == text[:512]
