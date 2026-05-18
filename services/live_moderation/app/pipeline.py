"""End-to-end live pipeline: VAD chunking -> ASR -> toxicity -> events.

This module is intentionally pure-Python and synchronous (no asyncio, no
websockets) so it's unit-testable in isolation. The websocket handler in
``main.py`` wraps it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .asr import WhisperASR
from .config import Settings
from .schemas import Action, FlagEvent, TranscriptEvent
from .toxicity import StreamingToxicity
from .vad import VadChunker


@dataclass
class LivePipeline:
    """Stateful per-stream pipeline. One instance per connected stream."""

    settings: Settings
    asr: WhisperASR
    toxicity: StreamingToxicity
    chunker: VadChunker

    @classmethod
    def for_settings(
        cls,
        settings: Settings,
        sample_rate: int,
        asr: WhisperASR,
        toxicity: StreamingToxicity,
    ) -> "LivePipeline":
        chunker = VadChunker(
            sample_rate=sample_rate,
            chunk_target_s=settings.chunk_target_s,
            silence_break_s=settings.silence_break_s,
            min_voiced_s=settings.min_voiced_s,
            energy_threshold=settings.vad_energy_threshold,
        )
        return cls(settings=settings, asr=asr, toxicity=toxicity, chunker=chunker)

    def push(self, samples: np.ndarray) -> Iterable[TranscriptEvent | FlagEvent]:
        """Push samples; yield transcript + flag events as chunks close."""
        for chunk in self.chunker.push(samples):
            yield from self._process_chunk(chunk)

    def flush(self) -> Iterable[TranscriptEvent | FlagEvent]:
        chunk = self.chunker.flush()
        if chunk is not None:
            yield from self._process_chunk(chunk)

    def _process_chunk(self, chunk):
        # ASR.
        asr_out = self.asr.transcribe(chunk.samples)
        text = asr_out.text
        yield TranscriptEvent(
            chunk_id=chunk.chunk_id,
            start_s=chunk.start_s,
            end_s=chunk.end_s,
            text=text,
        )

        # Toxicity.
        tox = self.toxicity.classify(text)
        if tox is None:
            return
        action = self._decide(tox.severity)
        if action == "NONE":
            return
        yield FlagEvent(
            chunk_id=chunk.chunk_id,
            start_s=chunk.start_s,
            end_s=chunk.end_s,
            severity=tox.severity,
            category=tox.category,
            action=action,
            explanation=tox.explanation,
        )

    def _decide(self, severity: float) -> Action:
        if severity >= self.settings.block_above:
            return "BLOCK"
        if severity >= self.settings.warn_above:
            return "WARN"
        return "NONE"
