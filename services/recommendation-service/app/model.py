"""LightFM-based collaborative filtering model.

Trains a LightFM warp model nightly from watch-events pulled from metadata-service.
Predictions for a given user are cached in Redis for fast `/recommend/{user_id}` reads.

Cold-start users fall back to popularity-ranked recent videos.
"""
import json
import logging
import pickle
from typing import List

import httpx
import numpy as np
import redis
from lightfm import LightFM
from scipy.sparse import coo_matrix

from .config import settings

log = logging.getLogger("rec.model")

_R = redis.Redis.from_url(settings.redis_url)
MODEL_BLOB_KEY = "rec:model"
USERS_KEY = "rec:users"
ITEMS_KEY = "rec:items"


def _fetch_events() -> list[dict]:
    r = httpx.get(f"{settings.metadata_url}/watch-events", params={"limit": 50000}, timeout=60.0)
    r.raise_for_status()
    return r.json()


def _fetch_popular(limit: int = 50) -> List[str]:
    r = httpx.get(f"{settings.metadata_url}/videos",
                  params={"status": "ready", "limit": limit}, timeout=30.0)
    r.raise_for_status()
    return [v["id"] for v in r.json()]


def train() -> dict:
    events = _fetch_events()
    if not events:
        return {"trained": False, "reason": "no events"}

    users = sorted({e["user_id"] for e in events})
    items = sorted({e["video_id"] for e in events})
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {v: i for i, v in enumerate(items)}

    rows, cols, data = [], [], []
    for e in events:
        rows.append(u_idx[e["user_id"]])
        cols.append(i_idx[e["video_id"]])
        weight = 1.0 + (2.0 if e.get("completed") else 0.0) + min(e["watched_seconds"] / 60.0, 5.0)
        data.append(weight)

    mat = coo_matrix((data, (rows, cols)), shape=(len(users), len(items)))
    model = LightFM(loss="warp", no_components=32, learning_rate=0.05)
    model.fit(mat, epochs=10, num_threads=2)

    _R.set(MODEL_BLOB_KEY, pickle.dumps(model))
    _R.set(USERS_KEY, json.dumps(users))
    _R.set(ITEMS_KEY, json.dumps(items))
    log.info("Trained LightFM: %d users, %d items, %d events", len(users), len(items), len(events))
    return {"trained": True, "users": len(users), "items": len(items)}


def recommend(user_id: str, top_k: int = 20) -> List[str]:
    blob = _R.get(MODEL_BLOB_KEY)
    users_raw = _R.get(USERS_KEY)
    items_raw = _R.get(ITEMS_KEY)
    if not (blob and users_raw and items_raw):
        return _fetch_popular(top_k)
    users = json.loads(users_raw)
    items = json.loads(items_raw)
    if user_id not in users:
        return _fetch_popular(top_k)
    model: LightFM = pickle.loads(blob)
    u_i = users.index(user_id)
    scores = model.predict(u_i, np.arange(len(items)))
    order = np.argsort(-scores)
    return [items[i] for i in order[:top_k]]
