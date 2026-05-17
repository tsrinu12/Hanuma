"""Distributed cache with stampede protection.

When many requests hit a cold cache for the same key simultaneously (cache
stampede / thundering herd), naive code regenerates the value once per
request. We solve this with a short-lived Redis SETNX lock: the first
requester computes, everyone else either waits briefly or returns a stale
value.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from typing import Any, Callable, Optional

import redis

log = logging.getLogger("opt.redis_cache")


def _key_hash(args: tuple, kwargs: dict) -> str:
    payload = json.dumps([args, sorted(kwargs.items())], default=repr, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def cached(
    redis_client: redis.Redis,
    *,
    namespace: str,
    ttl: int = 300,
    lock_timeout: int = 10,
    encoder=json.dumps,
    decoder=json.loads,
):
    """Distributed cache decorator with stampede protection.

    Usage:
        @cached(r, namespace="video_meta", ttl=60)
        def get_video(video_id: str) -> dict: ...
    """
    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            k = f"cache:{namespace}:{_key_hash(args, kwargs)}"
            lock_k = k + ":lock"

            hit = redis_client.get(k)
            if hit is not None:
                try:
                    return decoder(hit)
                except Exception:
                    pass  # corrupted entry, recompute

            # Try to acquire compute-lock; if someone else has it, wait briefly
            # then read again (single-flight pattern).
            got = redis_client.set(lock_k, "1", nx=True, ex=lock_timeout)
            if not got:
                for _ in range(20):
                    time.sleep(0.05)
                    hit = redis_client.get(k)
                    if hit is not None:
                        return decoder(hit)
                # If we waited and still no value, fall through and compute ourselves

            try:
                value = fn(*args, **kwargs)
                redis_client.setex(k, ttl, encoder(value))
                return value
            finally:
                redis_client.delete(lock_k)
        return wrapper
    return deco
