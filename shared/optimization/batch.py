"""Latency-bounded micro-batcher for GPU inference.

Pattern: incoming HTTP requests post their input to a shared queue; a single
worker thread drains the queue in batches of up to `max_batch_size`, with a
hard wait ceiling of `max_wait_ms`. This shrinks GPU latency dramatically when
many concurrent users hit the same model — one forward pass services 16
requests instead of 16 forward passes.

Each caller blocks on a Future until its slot in the batch is processed.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Callable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")
log = logging.getLogger("opt.batch")


class MicroBatcher:
    """Generic micro-batcher. `infer_batch` takes a list of inputs and returns
    a list of outputs of the same length, in the same order."""

    def __init__(
        self,
        infer_batch: Callable[[List], List],
        *,
        max_batch_size: int = 16,
        max_wait_ms: int = 25,
    ):
        self._infer = infer_batch
        self.max_batch = max_batch_size
        self.max_wait = max_wait_ms / 1000.0
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=2.0)

    def submit(self, item) -> Future:
        fut: Future = Future()
        self._q.put((item, fut))
        return fut

    def _drain_once(self):
        batch_items: List = []
        batch_futs: List[Future] = []
        deadline = None
        while len(batch_items) < self.max_batch:
            timeout = None
            if deadline is not None:
                timeout = max(0.0, deadline - time.monotonic())
                if timeout == 0.0:
                    break
            try:
                item, fut = self._q.get(timeout=timeout)
            except queue.Empty:
                break
            batch_items.append(item)
            batch_futs.append(fut)
            if deadline is None:
                deadline = time.monotonic() + self.max_wait
        return batch_items, batch_futs

    def _loop(self):
        while not self._stop.is_set():
            try:
                # Block until at least one item arrives
                first = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            item, fut = first
            batch_items = [item]
            batch_futs = [fut]
            deadline = time.monotonic() + self.max_wait
            while len(batch_items) < self.max_batch:
                timeout = max(0.0, deadline - time.monotonic())
                if timeout == 0.0:
                    break
                try:
                    item, fut = self._q.get(timeout=timeout)
                    batch_items.append(item)
                    batch_futs.append(fut)
                except queue.Empty:
                    break
            try:
                results = self._infer(batch_items)
                if len(results) != len(batch_items):
                    raise RuntimeError(f"infer_batch returned {len(results)} results for {len(batch_items)} inputs")
                for f, r in zip(batch_futs, results):
                    f.set_result(r)
            except Exception as e:
                log.exception("batch infer failed")
                for f in batch_futs:
                    f.set_exception(e)
