"""Circuit breaker for upstream HTTP / RPC calls.

States:
  CLOSED   — calls flow normally; track failures
  OPEN     — short-circuit: raise immediately for `recovery_timeout` seconds
  HALF_OPEN — allow one trial call; if it succeeds, close; if it fails, re-open

Use this to wrap calls between services so a slow / down downstream doesn't
take down the upstream too.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

log = logging.getLogger("opt.circuit")


class CircuitOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._lock = threading.Lock()
        self._state = "CLOSED"
        self._failures = 0
        self._opened_at = 0.0

    @property
    def state(self) -> str:
        return self._state

    def _maybe_recover(self) -> None:
        if self._state == "OPEN" and time.time() - self._opened_at >= self.recovery_timeout:
            self._state = "HALF_OPEN"
            log.info("circuit %s -> HALF_OPEN", self.name)

    def call(self, fn: Callable, *args, **kwargs):
        with self._lock:
            self._maybe_recover()
            if self._state == "OPEN":
                raise CircuitOpenError(f"circuit {self.name} is OPEN")

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result

    def _on_success(self):
        with self._lock:
            self._failures = 0
            if self._state in ("HALF_OPEN", "OPEN"):
                log.info("circuit %s -> CLOSED", self.name)
            self._state = "CLOSED"

    def _on_failure(self):
        with self._lock:
            self._failures += 1
            if self._state == "HALF_OPEN" or self._failures >= self.failure_threshold:
                self._state = "OPEN"
                self._opened_at = time.time()
                log.warning("circuit %s -> OPEN (failures=%d)", self.name, self._failures)
