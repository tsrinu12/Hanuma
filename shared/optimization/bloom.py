"""Tiny Bloom filter — used by content_dedupe and the "seen this video today"
counter in the recommendation service.

Implements a fixed-size bitset with k MurmurHash3 hash functions. False-positive
rate is configurable via capacity + bytes; default settings give ~1% FPR at 1M
items with ~1.2 MB memory.
"""
from __future__ import annotations

import hashlib
import math
from typing import Iterable


class BloomFilter:
    def __init__(self, capacity: int, error_rate: float = 0.01):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        # Standard formulas
        self.capacity = capacity
        self.error_rate = error_rate
        m = -(capacity * math.log(error_rate)) / (math.log(2) ** 2)
        k = (m / capacity) * math.log(2)
        self.num_bits = int(math.ceil(m))
        self.num_hashes = max(1, int(math.ceil(k)))
        self._bits = bytearray((self.num_bits + 7) // 8)

    @classmethod
    def from_bytes(cls, blob: bytes, capacity: int, error_rate: float = 0.01):
        inst = cls(capacity, error_rate)
        # Truncate / pad to match the expected size
        n = len(inst._bits)
        inst._bits[: min(n, len(blob))] = blob[:n]
        return inst

    def to_bytes(self) -> bytes:
        return bytes(self._bits)

    # ---- hashing -------------------------------------------------------

    def _hashes(self, item: str) -> Iterable[int]:
        b = item.encode("utf-8") if isinstance(item, str) else bytes(item)
        h1 = int.from_bytes(hashlib.md5(b).digest()[:8], "big")
        h2 = int.from_bytes(hashlib.sha1(b).digest()[:8], "big")
        # Double-hashing trick (Kirsch-Mitzenmacher) — generate k hashes from 2
        for i in range(self.num_hashes):
            yield (h1 + i * h2) % self.num_bits

    # ---- ops ----------------------------------------------------------

    def add(self, item: str) -> None:
        for h in self._hashes(item):
            self._bits[h >> 3] |= 1 << (h & 7)

    def contains(self, item: str) -> bool:
        for h in self._hashes(item):
            if not (self._bits[h >> 3] & (1 << (h & 7))):
                return False
        return True

    __contains__ = contains
