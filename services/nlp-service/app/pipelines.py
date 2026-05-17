"""Lazy-loaded NLP pipelines.

We hold one ClassifierBundle per process; pipelines are loaded on first use so
the service starts quickly even with cold model caches.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from .config import settings

log = logging.getLogger("nlp.pipelines")


class NLPBundle:
    def __init__(self) -> None:
        self.device = 0 if (settings.device == "cuda" and torch.cuda.is_available()) else -1
        log.info("Loading NLP pipelines on device=%s", self.device)
        self.sentiment = pipeline(
            "text-classification", model=settings.sentiment_model, device=self.device,
            top_k=None, truncation=True,
        )
        self.ner = pipeline(
            "token-classification", model=settings.ner_model, device=self.device,
            aggregation_strategy="simple",
        )
        self.emotion = pipeline(
            "text-classification", model=settings.emotion_model, device=self.device,
            top_k=None, truncation=True,
        )
        self.summarizer = pipeline(
            "summarization", model=settings.summary_model, device=self.device,
        )
        self.embedder = SentenceTransformer(settings.embedding_model)
        try:
            from keybert import KeyBERT  # type: ignore

            self._keybert = KeyBERT(model=self.embedder)
        except Exception as e:  # pragma: no cover
            log.warning("KeyBERT unavailable: %s", e)
            self._keybert = None
        try:
            from bertopic import BERTopic  # type: ignore

            self._bertopic_cls = BERTopic
        except Exception as e:  # pragma: no cover
            log.warning("BERTopic unavailable: %s", e)
            self._bertopic_cls = None

    # ---- text-level analyses --------------------------------------------

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        out = self.sentiment(text)[0]
        return {x["label"]: float(x["score"]) for x in out}

    def analyze_emotion(self, text: str) -> Dict[str, float]:
        out = self.emotion(text)[0]
        return {x["label"]: float(x["score"]) for x in out}

    def extract_entities(self, text: str) -> List[Dict]:
        ents = self.ner(text)
        return [
            {
                "text": e["word"],
                "type": e["entity_group"],
                "start": int(e["start"]),
                "end": int(e["end"]),
                "score": float(e["score"]),
            }
            for e in ents
        ]

    def keyphrases(self, text: str, top_n: int = 10) -> List[Dict]:
        if self._keybert is None:
            return []
        kw = self._keybert.extract_keywords(
            text, top_n=top_n, keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.5
        )
        return [{"phrase": p, "score": float(s)} for p, s in kw]

    # ---- corpus-level analyses ------------------------------------------

    def topics(self, docs: List[str]) -> List[Dict]:
        if self._bertopic_cls is None or len(docs) < 3:
            return []
        model = self._bertopic_cls(
            embedding_model=self.embedder,
            nr_topics=min(settings.max_topics, max(2, len(docs) // 3)),
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _ = model.fit_transform(docs)
        info = model.get_topic_info()
        out: List[Dict] = []
        for _, row in info.iterrows():
            tid = int(row["Topic"])
            if tid == -1:
                continue
            terms = model.get_topic(tid) or []
            out.append({
                "topic_id": tid,
                "size": int(row["Count"]),
                "terms": [{"term": t, "weight": float(w)} for t, w in terms[:10]],
            })
        return out

    def summarize(self, text: str, max_length: int = 180) -> str:
        out = self.summarizer(
            text, max_length=max_length, min_length=max(32, max_length // 4), do_sample=False,
        )
        return out[0]["summary_text"].strip()
