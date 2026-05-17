"""In-process LRU + TTL caches.

Decorator wrappers around `cachetools` so services can mark a function as
"cache results for N seconds, evict least-recently-used beyond M entries"
without rolling their own.

These are NOT distributed — for cross-replica caching use redis_cache.
"""
from __future__ import annotations

import functools
import hashlib
import json
import time
from typing import Any, Callable, Optional, Tuple

try:
    from cachetools import LRUCache, TTLCache  # type: ignore
except ImportError:  # pragma: no cover
    LRUCache = TTLCache = None  # type: ignore


def _key(args: tuple, kwargs: dict) -> str:
    # Stable hash of arguments — handles unhashable kwargs by JSON-encoding
    try:
        return json.dumps([args, sorted(kwargs.items())], default=repr, sort_keys=True)
    except Exception:
        return repr((args, kwargs))


def lru(maxsize: int = 1024):
    """`@lru(maxsize=512)` — simple LRU, no TTL."""
    if LRUCache is None:
        raise RuntimeError("install cachetools")
    cache: Any = LRUCache(maxsize=maxsize)

    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            k = _key(args, kwargs)
            if k in cache:
                return cache[k]
            v = fn(*args, **kwargs)
            cache[k] = v
            return v
        wrapper.cache_clear = cache.clear  # type: ignore
        return wrapper
    return deco


def ttl(maxsize: int = 1024, seconds: int = 60):
    """`@ttl(maxsize=512, seconds=30)` — LRU + per-entry TTL."""
    if TTLCache is None:
        raise RuntimeError("install cachetools")
    cache: Any = TTLCache(maxsize=maxsize, ttl=seconds)

    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            k = _key(args, kwargs)
            if k in cache:
                return cache[k]
            v = fn(*args, **kwargs)
            cache[k] = v
            return v
        wrapper.cache_clear = cache.clear  # type: ignore
        return wrapper
    return deco
