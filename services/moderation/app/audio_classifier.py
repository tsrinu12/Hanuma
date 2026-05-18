"""Audio moderation.

Lite mode: a feature-engineered rule classifier over five precomputed
features per audio segment:
    [mean_db, std_db, peak_db, percussive_ratio, voicing_ratio]

  - Spike in peak_db with low voicing_ratio -> "loud non-speech burst"
    (proxy for gunshot / explosion in lite mode).
  - High percussive_ratio + high peak_db -> "violent action audio".
  - All loudness very high persistently -> "shouting / screaming".

This is obviously a placeholder. In prod we drop in PANNs CNN14 over a raw
waveform and use its 527-class AudioSet head, then map AudioSet categories
into our seven-class vocabulary (Gunshot, Screaming, Explosion, etc.).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from .schemas import AudioSegment, ModalityFlag


logger = logging.getLogger(__name__)


# Indices into the 5-feature vector
F_MEAN_DB, F_STD_DB, F_PEAK_DB, F_PERCUSSIVE, F_VOICING = range(5)


@dataclass
class AudioClassifier:
    profile: str = "lite"
    _model = None
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self.profile == "lite" or self._model is not None

    def _load_prod(self) -> None:
        # Placeholder. In a real prod build:
        #   from panns_inference import AudioTagging
        #   self._model = AudioTagging(checkpoint_path=..., device="cuda")
        with self._lock:
            if self._model is not None:
                return
            logger.warning(
                "audio prod model not wired; falling back to lite rules"
            )
            self._model = "lite-fallback"

    def _classify_lite(self, segments: list[AudioSegment]) -> list[ModalityFlag]:
        flags: list[ModalityFlag] = []
        for seg in segments:
            f = seg.features
            if f is None or len(f) < 5:
                continue
            mean_db, std_db, peak_db, percussive, voicing = f[:5]

            # Loud non-speech burst.
            if peak_db > -3.0 and voicing < 0.2:
                flags.append(
                    ModalityFlag(
                        modality="audio",
                        category="violence",
                        severity=min(1.0, (peak_db + 6.0) / 6.0),
                        start_s=seg.start_s,
                        end_s=seg.end_s,
                        explanation=(
                            "audio: loud non-speech burst (proxy for gunshot/"
                            "explosion)"
                        ),
                    )
                )
                continue

            # Sustained loud + percussive: violent action audio.
            if percussive > 0.6 and peak_db > -6.0:
                flags.append(
                    ModalityFlag(
                        modality="audio",
                        category="violence",
                        severity=min(1.0, percussive),
                        start_s=seg.start_s,
                        end_s=seg.end_s,
                        explanation="audio: percussive + loud (action audio)",
                    )
                )
                continue

            # Shouting / screaming: very loud, high voicing.
            if mean_db > -8.0 and voicing > 0.7:
                flags.append(
                    ModalityFlag(
                        modality="audio",
                        category="harassment",
                        severity=min(1.0, (mean_db + 12.0) / 12.0),
                        start_s=seg.start_s,
                        end_s=seg.end_s,
                        explanation="audio: shouting / screaming",
                    )
                )
        return flags

    def classify(self, segments: list[AudioSegment]) -> list[ModalityFlag]:
        if self.profile == "lite":
            return self._classify_lite(segments)
        self._load_prod()
        # For now, prod path also falls back to lite rules.
        return self._classify_lite(segments)
