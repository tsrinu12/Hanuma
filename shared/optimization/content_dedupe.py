"""Content-hash deduplication for uploads.

If a user re-uploads the same video file, we don't want to pay the FFmpeg +
Whisper + V-JEPA cost again. We hash the raw bytes during upload, look up the
hash in Redis, and if we've seen it before we just associate the new video_id
with the existing transcoded output.

The Bloom filter is the cheap fast-path: 99% of the time it correctly says
"never seen this" and we proceed to transcode. The 1% it says "maybe seen" we
fall through to Redis for the authoritative check.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import IO, Iterable, Optional

import redis

from .bloom import BloomFilter

log = logging.getLogger("opt.content_dedupe")


@dataclass
class DedupeHit:
    sha256: str
    original_video_id: str
    s3_hls_prefix: str


class ContentDedupe:
    """Stream-friendly hasher + Redis-backed dedupe registry."""

    REDIS_KEY = "dedupe:sha256:"   # value = json{video_id, s3_hls_prefix, ts}
    BLOOM_KEY = "dedupe:bloom"     # bytes-blob bitset, regenerated weekly

    def __init__(self, redis_client: redis.Redis, bloom_capacity: int = 1_000_000):
        self._r = redis_client
        # Local in-process bloom — refreshed from Redis on instantiation
        raw = self._r.get(self.BLOOM_KEY)
        if raw:
            self.bloom = BloomFilter.from_bytes(raw, bloom_capacity)
        else:
            self.bloom = BloomFilter(bloom_capacity)

    # ---- hashing -------------------------------------------------------

    @staticmethod
    def hash_stream(chunks: Iterable[bytes]) -> str:
        h = hashlib.sha256()
        for c in chunks:
            h.update(c)
        return h.hexdigest()

    @staticmethod
    def hash_file_handle(fh: IO[bytes], chunk_size: int = 1 << 20) -> str:
        h = hashlib.sha256()
        while True:
            c = fh.read(chunk_size)
            if not c:
                break
            h.update(c)
        return h.hexdigest()

    # ---- lookup / register --------------------------------------------

    def lookup(self, sha256: str) -> Optional[DedupeHit]:
        if not self.bloom.contains(sha256):
            return None  # definitely not seen
        raw = self._r.get(self.REDIS_KEY + sha256)
        if raw is None:
            return None  # bloom false-positive
        d = json.loads(raw)
        return DedupeHit(sha256=sha256, **d)

    def register(self, sha256: str, video_id: str, s3_hls_prefix: str) -> None:
        self.bloom.add(sha256)
        self._r.set(
            self.REDIS_KEY + sha256,
            json.dumps({"original_video_id": video_id, "s3_hls_prefix": s3_hls_prefix}),
        )
        # Persist updated bloom bytes (cheap; bloom is bounded size)
        self._r.set(self.BLOOM_KEY, self.bloom.to_bytes())
        log.info("dedupe register sha=%s video=%s", sha256[:12], video_id)
