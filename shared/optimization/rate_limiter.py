"""Token-bucket rate limiter.

Two flavours:
  - LocalTokenBucket  — per-process (use for per-replica throttling)
  - RedisTokenBucket  — per-cluster (use for per-user / per-API-key limits)

The Redis version uses Lua to make refill + take atomic so concurrent requests
can't steal more than `capacity` tokens.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import redis


@dataclass
class LocalTokenBucket:
    rate_per_sec: float
    capacity: int

    def __post_init__(self):
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def take(self, n: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate_per_sec)
            self._last = now
            if self._tokens >= n:
                self._tokens -= n
                return True
            return False


_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local cap  = tonumber(ARGV[2])
local now  = tonumber(ARGV[3])
local n    = tonumber(ARGV[4])

local bucket = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(bucket[1])
local last   = tonumber(bucket[2])
if tokens == nil then
  tokens = cap
  last   = now
end
local delta = math.max(0, now - last) * rate
tokens = math.min(cap, tokens + delta)
local allowed = 0
if tokens >= n then
  tokens = tokens - n
  allowed = 1
end
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, math.ceil(cap / rate) * 2)
return allowed
"""


class RedisTokenBucket:
    def __init__(self, redis_client: redis.Redis, *, rate_per_sec: float, capacity: int):
        self._r = redis_client
        self.rate = rate_per_sec
        self.cap = capacity
        self._script = redis_client.register_script(_LUA)

    def take(self, key: str, n: int = 1) -> bool:
        allowed = self._script(keys=["rl:" + key], args=[self.rate, self.cap, time.time(), n])
        return bool(allowed)
