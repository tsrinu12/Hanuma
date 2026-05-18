"""Per-chunk toxicity classifier for live streams.

Lite mode: same rule set as the offline ``moderation`` service text
classifier, but stripped to a single (category, severity, explanation)
output per chunk for low-latency streaming.

Prod mode: toxic-bert (lazy-loaded).
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass


@dataclass
class ToxicityResult:
    category: str
    severity: float
    explanation: str


# Same family as services/moderation/app/text_classifier.py but compressed.
_RULES: list[tuple[str, re.Pattern, float, str]] = [
    ("harassment", re.compile(r"\b(go\s*kill\s*yourself|kys)\b", re.I), 0.95, "explicit harassment"),
    ("violence",   re.compile(r"\b(kill\s+(him|her|them)|murder|shoot\s+(him|her|them))\b", re.I), 0.92, "violence threat"),
    ("toxicity",   re.compile(r"\b(idiot|moron|stupid)\b", re.I), 0.55, "personal insult"),
    ("hate",       re.compile(r"\b(racial slur placeholder)\b", re.I), 0.90, "hate term"),
    ("self_harm",  re.compile(r"\b(cut\s+myself|end\s+my\s+life)\b", re.I), 0.85, "self-harm reference"),
]


class StreamingToxicity:
    def __init__(self, profile: str = "lite", model_name: str = "unitary/toxic-bert") -> None:
        self.profile = profile
        self.model_name = model_name
        self._model = None
        self._tok = None
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self.profile == "lite" or self._model is not None

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
            except ImportError as e:
                raise RuntimeError("transformers not installed for prod") from e
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()

    def classify(self, text: str) -> ToxicityResult | None:
        text = text.strip()
        if not text:
            return None
        if self.profile == "lite":
            return self._classify_lite(text)
        return self._classify_prod(text)

    # --- lite ---

    def _classify_lite(self, text: str) -> ToxicityResult | None:
        best: ToxicityResult | None = None
        for category, pat, severity, explanation in _RULES:
            if pat.search(text):
                if best is None or severity > best.severity:
                    best = ToxicityResult(
                        category=category, severity=severity, explanation=explanation
                    )
        return best

    # --- prod ---

    _LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    def _classify_prod(self, text: str) -> ToxicityResult | None:
        import torch  # type: ignore

        self._load()
        with torch.no_grad():
            tok = self._tok(text, padding=True, truncation=True, max_length=128, return_tensors="pt")  # type: ignore[union-attr]
            scores = torch.sigmoid(self._model(**tok).logits)[0].cpu().numpy()  # type: ignore[union-attr]

        best: ToxicityResult | None = None
        for label, s in zip(self._LABELS, scores):
            if float(s) < 0.5:
                continue
            cat = {
                "toxic": "toxicity",
                "severe_toxic": "toxicity",
                "obscene": "toxicity",
                "threat": "violence",
                "insult": "harassment",
                "identity_hate": "hate",
            }.get(label, "toxicity")
            r = ToxicityResult(category=cat, severity=float(s), explanation=f"toxic-bert: {label}")
            if best is None or r.severity > best.severity:
                best = r
        return best
