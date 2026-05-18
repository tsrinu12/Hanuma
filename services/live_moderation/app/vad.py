"""Voice-activity-detection chunker.

Frames in: raw 16-bit PCM bytes (or float32 array). Frame rate is
``sample_rate``. We accumulate frames into chunks based on:

  1. Speech detected (RMS > energy_threshold)
  2. Either ``chunk_target_s`` of speech accumulated, OR
     ``silence_break_s`` of silence after at least ``min_voiced_s`` of speech

When a chunk closes we yield ``(start_s, end_s, samples_float32)``.

Lite VAD: RMS energy threshold. Cheap, dependency-free, surprisingly
serviceable for podcast/talking-head streams. Prod swaps in Silero VAD via
``silero-vad`` for better robustness to background noise.

The chunker is stateful; create one per stream.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    chunk_id: int
    start_s: float
    end_s: float
    samples: np.ndarray            # float32 mono in [-1, 1]


class VadChunker:
    def __init__(
        self,
        sample_rate: int,
        chunk_target_s: float,
        silence_break_s: float,
        min_voiced_s: float,
        energy_threshold: float,
    ) -> None:
        self.sr = int(sample_rate)
        self.target_n = int(chunk_target_s * sample_rate)
        self.silence_break_n = int(silence_break_s * sample_rate)
        self.min_voiced_n = int(min_voiced_s * sample_rate)
        self.energy_threshold = float(energy_threshold)

        self._buf: deque[np.ndarray] = deque()
        self._buf_samples = 0
        # Sample indices, used to compute timestamps.
        self._chunk_start_idx: int | None = None
        self._global_idx = 0
        self._silence_run = 0
        self._voiced_in_chunk = 0
        self._next_chunk_id = 0

    # --- public ---

    def push(self, samples: np.ndarray) -> list[Chunk]:
        """Push one batch of float32 samples; return any chunks that closed."""
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        samples = np.asarray(samples, dtype=np.float32)
        out: list[Chunk] = []

        # We process in small windows so VAD decisions are fine-grained.
        win = max(1, self.sr // 50)   # 20 ms windows
        for start in range(0, len(samples), win):
            window = samples[start : start + win]
            if window.size == 0:
                continue
            rms = float(np.sqrt(np.mean(window * window) + 1e-12))
            voiced = rms > self.energy_threshold

            if self._chunk_start_idx is None and voiced:
                self._chunk_start_idx = self._global_idx + start
                self._silence_run = 0
                self._voiced_in_chunk = 0
                self._buf.clear()
                self._buf_samples = 0

            if self._chunk_start_idx is not None:
                self._buf.append(window)
                self._buf_samples += window.size
                if voiced:
                    self._voiced_in_chunk += window.size
                    self._silence_run = 0
                else:
                    self._silence_run += window.size

                # Close conditions.
                close = False
                if self._buf_samples >= self.target_n:
                    close = True
                elif (
                    self._silence_run >= self.silence_break_n
                    and self._voiced_in_chunk >= self.min_voiced_n
                ):
                    close = True

                if close:
                    chunk = self._emit_chunk()
                    if chunk is not None:
                        out.append(chunk)

        self._global_idx += len(samples)
        return out

    def flush(self) -> Chunk | None:
        """Force-close any in-progress chunk. Call when the stream ends."""
        if self._chunk_start_idx is None:
            return None
        if self._voiced_in_chunk < self.min_voiced_n:
            # Not enough speech; drop.
            self._reset()
            return None
        return self._emit_chunk()

    # --- internals ---

    def _emit_chunk(self) -> Chunk | None:
        if self._chunk_start_idx is None:
            return None
        samples = np.concatenate(list(self._buf)) if self._buf else np.zeros(0, dtype=np.float32)
        start_s = self._chunk_start_idx / self.sr
        end_s = (self._chunk_start_idx + samples.size) / self.sr
        chunk = Chunk(
            chunk_id=self._next_chunk_id,
            start_s=start_s,
            end_s=end_s,
            samples=samples,
        )
        self._next_chunk_id += 1
        self._reset()
        return chunk

    def _reset(self) -> None:
        self._buf.clear()
        self._buf_samples = 0
        self._chunk_start_idx = None
        self._silence_run = 0
        self._voiced_in_chunk = 0
