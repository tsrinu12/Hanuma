"""Transformer-based abuse / hate / spam classification.

We run three heads:
  - toxic-bert         : multi-label toxic / obscene / insult / threat / identity_hate
  - dehatebert         : binary hate-speech
  - zero-shot mDeBERTa : open-ended labels (spam, advertisement, scam, phishing,
                          self-promotion, fan_engagement). This lets the
                          classifier handle marketing-style spam that toxic-bert
                          doesn't flag.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import torch
from transformers import pipeline

from .config import settings

log = logging.getLogger("spam.classifiers")

SPAM_LABELS = [
    "spam",
    "advertisement",
    "scam or phishing",
    "self-promotion",
    "legitimate comment",
]


class ClassifierBundle:
    def __init__(self) -> None:
        self.device = 0 if (settings.device == "cuda" and torch.cuda.is_available()) else -1
        log.info("Loading toxic-bert / dehatebert / zero-shot on device=%s", self.device)
        self.toxic = pipeline(
            "text-classification",
            model=settings.toxic_model,
            top_k=None,
            device=self.device,
        )
        self.hate = pipeline(
            "text-classification",
            model=settings.hate_model,
            device=self.device,
        )
        self.zeroshot = pipeline(
            "zero-shot-classification",
            model=settings.zeroshot_model,
            device=self.device,
        )

    def score(self, text: str) -> Dict[str, Dict[str, float]]:
        toxic_out = self.toxic(text, truncation=True)[0]
        toxic_scores = {x["label"]: float(x["score"]) for x in toxic_out}

        hate_out = self.hate(text, truncation=True)[0]
        hate_score = (
            float(hate_out["score"]) if hate_out["label"].upper().startswith(("HATE", "LABEL_1")) else 1 - float(hate_out["score"])
        )

        zs = self.zeroshot(text, candidate_labels=SPAM_LABELS, multi_label=False)
        zs_scores = {lab: float(s) for lab, s in zip(zs["labels"], zs["scores"])}

        return {
            "toxic": toxic_scores,
            "hate": {"hate_speech": hate_score},
            "spam_intent": zs_scores,
        }
