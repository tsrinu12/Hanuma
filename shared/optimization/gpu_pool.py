"""GPU concurrency gate.

A semaphore that caps the number of in-flight GPU calls per replica.
Combined with MicroBatcher, this prevents OOMs on bursty traffic — extra
callers wait in line instead of all trying to allocate VRAM at once.
"""
from __future__ import annotations

import contextlib
import os
import threading
from typing import Optional


class GPUPool:
    def __init__(self, max_concurrency: Optional[int] = None):
        if max_concurrency is None:
            max_concurrency = int(os.getenv("GPU_MAX_CONCURRENCY", "2"))
        self._sem = threading.BoundedSemaphore(max_concurrency)
        self.max_concurrency = max_concurrency

    @contextlib.contextmanager
    def slot(self, timeout: Optional[float] = None):
        acquired = self._sem.acquire(timeout=timeout) if timeout else self._sem.acquire()
        if not acquired:
            raise TimeoutError("GPU pool exhausted")
        try:
            yield
        finally:
            self._sem.release()
