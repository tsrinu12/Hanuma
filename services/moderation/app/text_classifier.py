"""Text moderation.

Lite mode: deterministic keyword + pattern rules, mapped onto the same
``(category, severity)`` schema as the prod model. This is good enough to
exercise the API and fusion logic, but obviously not production quality.

Prod mode: ``unitary/toxic-bert`` — a multi-label classifier with categories
{toxic, severe_toxic, obscene, threat, insult, identity_hate}. We map those
onto our seven-category vocabulary.

The classifier returns one ``ModalityFlag`` per transcript segment that
crosses any per-category threshold, so fusion can see *which* segment fired.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from .schemas import Category, ModalityFlag, TranscriptSegment


logger = logging.getLogger(__name__)


# Conservative lite-mode rule set. Designed for tests, not for moderation.
_RULES: list[tuple[Category, re.Pattern, float, str]] = [
    ("toxicity",   re.compile(r"\b(idiot|moron|stupid)\b", re.I),      0.55, "use of personal insult"),
    ("harassment", re.compile(r"\b(go\s*kill\s*yourself|kys)\b", re.I), 0.95, "explicit harassment / suicide instruction"),
    ("violence",   re.compile(r"\b(kill\s+(him|her|them)|murder|shoot\s+(him|her|them))\b", re.I), 0.92, "explicit violence threat"),
    ("hate",       re.compile(r"\b(racial slur placeholder)\b", re.I),  0.90, "hate speech term"),
    ("sexual",     re.compile(r"\b(explicit\s+sexual\s+act)\b", re.I),  0.80, "explicit sexual reference"),
    ("self_harm",  re.compile(r"\b(cut\s+myself|end\s+my\s+life)\b", re.I), 0.85, "self-harm reference"),
    ("spam",       re.compile(r"http[s]?://(buy|cheap|free)[a-z0-9.-]+", re.I), 0.40, "promotional URL"),
]


@dataclass
class TextClassifier:
    """Lite + Prod text classifier."""

    profile: str = "lite"
    model_name: str = "unitary/toxic-bert"
    _model = None
    _tok = None
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self.profile == "lite" or self._model is not None

    def _load_prod(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "transformers not installed for MODEL_PROFILE=prod"
                ) from e
            logger.info("loading text classifier: %s", self.model_name)
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()

    # --- lite ---

    def _classify_lite(self, segments: list[TranscriptSegment]) -> list[ModalityFlag]:
        flags: list[ModalityFlag] = []
        for seg in segments:
            for category, pat, severity, explanation in _RULES:
                if pat.search(seg.text):
                    flags.append(
                        ModalityFlag(
                            modality="text",
                            category=category,
                            severity=severity,
                            start_s=seg.start_s,
                            end_s=seg.end_s,
                            explanation=f"transcript: {explanation}",
                        )
                    )
        return flags

    # --- prod ---

    _TOXIC_BERT_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def _classify_prod(self, segments: list[TranscriptSegment]) -> list[ModalityFlag]:
        import torch  # type: ignore

        self._load_prod()
        flags: list[ModalityFlag] = []
        if not segments:
            return flags

        with torch.no_grad():
            tok = self._tok(  # type: ignore[union-attr]
                [s.text for s in segments],
                padding=True, truncation=True, max_length=128, return_tensors="pt",
            )
            logits = self._model(**tok).logits  # type: ignore[union-attr]
            scores = torch.sigmoid(logits).cpu().numpy()  # (n_seg, n_labels)

        for seg, row in zip(segments, scores):
            for label, score in zip(self._TOXIC_BERT_LABELS, row):
                if float(score) < 0.5:
                    continue
                category = self._map_label(label)
                flags.append(
                    ModalityFlag(
                        modality="text",
                        category=category,
                        severity=float(score),
                        start_s=seg.start_s,
                        end_s=seg.end_s,
                        explanation=f"transcript: toxic-bert flagged '{label}'",
                    )
                )
        return flags

    @staticmethod
    def _map_label(label: str) -> Category:
        return {
            "toxic": "toxicity",
            "severe_toxic": "toxicity",
            "obscene": "toxicity",
            "threat": "violence",
            "insult": "harassment",
            "identity_hate": "hate",
        }.get(label, "toxicity")  # type: ignore[return-value]

    # --- public ---

    def classify(self, segments: list[TranscriptSegment]) -> list[ModalityFlag]:
        if self.profile == "lite":
            return self._classify_lite(segments)
        return self._classify_prod(segments)
