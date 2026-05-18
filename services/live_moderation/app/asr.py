"""Streaming-friendly ASR wrapper.

Prod mode: faster-whisper (`tiny.en` / `medium.en`). Each chunk is
transcribed independently for ~2-3s end-to-end latency.

Lite mode: deterministic mock transcriber that returns a single token
echoing the chunk RMS — enough to exercise the rest of the pipeline in
tests without requiring a model download.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    text: str
    language: str | None = None


class WhisperASR:
    def __init__(self, model_name: str, sample_rate: int, profile: str) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.profile = profile
        self._model = None
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self.profile == "lite" or self._model is not None

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "faster-whisper not installed; pip install faster-whisper "
                    "or set MODEL_PROFILE=lite"
                ) from e
            logger.info("loading faster-whisper: %s", self.model_name)
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def transcribe(self, samples: np.ndarray) -> TranscriptResult:
        if self.profile == "lite":
            return self._transcribe_lite(samples)
        self._load()
        segments, info = self._model.transcribe(  # type: ignore[union-attr]
            samples.astype(np.float32),
            beam_size=1,
            vad_filter=False,         # we already chunked
            language="en",
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return TranscriptResult(text=text, language=info.language)

    def _transcribe_lite(self, samples: np.ndarray) -> TranscriptResult:
        # Deterministic stand-in. Real Whisper would go here.
        # We return a fixed placeholder so the rest of the pipeline behaves
        # the same way it would in prod for tests.
        if samples.size == 0:
            return TranscriptResult(text="", language="en")
        rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
        return TranscriptResult(text=f"[mock-asr rms={rms:.3f}]", language="en")
