"""nlp-service: deep-learning NLP for the platform.

Endpoints:
  POST /sentiment    body: {text}                  -> {label, scores}
  POST /emotion      body: {text}                  -> {scores}
  POST /entities     body: {text}                  -> {entities: [...]}
  POST /keyphrases   body: {text, top_n}           -> {phrases: [...]}
  POST /topics       body: {docs: [str]}           -> {topics: [...]}
  POST /summarize    body: {text, max_length}      -> {summary}
  POST /analyze      body: {text}                  -> all single-doc fields combined
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .pipelines import NLPBundle

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nlp-service")

app = FastAPI(title="distrebute.com - NLP Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_bundle: Optional[NLPBundle] = None


def bundle() -> NLPBundle:
    global _bundle
    if _bundle is None:
        _bundle = NLPBundle()
    return _bundle


class TextReq(BaseModel):
    text: str


class KeyphraseReq(BaseModel):
    text: str
    top_n: int = 10


class TopicsReq(BaseModel):
    docs: List[str]


class SummarizeReq(BaseModel):
    text: str
    max_length: int = 180


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/sentiment")
def sentiment(req: TextReq):
    scores = bundle().analyze_sentiment(req.text)
    label = max(scores, key=scores.get)
    return {"label": label, "scores": scores}


@app.post("/emotion")
def emotion(req: TextReq):
    return {"scores": bundle().analyze_emotion(req.text)}


@app.post("/entities")
def entities(req: TextReq):
    return {"entities": bundle().extract_entities(req.text)}


@app.post("/keyphrases")
def keyphrases(req: KeyphraseReq):
    return {"phrases": bundle().keyphrases(req.text, top_n=req.top_n)}


@app.post("/topics")
def topics(req: TopicsReq):
    return {"topics": bundle().topics(req.docs)}


@app.post("/summarize")
def summarize(req: SummarizeReq):
    return {"summary": bundle().summarize(req.text, max_length=req.max_length)}


@app.post("/analyze")
def analyze(req: TextReq):
    b = bundle()
    sent = b.analyze_sentiment(req.text)
    return {
        "sentiment": {"label": max(sent, key=sent.get), "scores": sent},
        "emotion": b.analyze_emotion(req.text),
        "entities": b.extract_entities(req.text),
        "keyphrases": b.keyphrases(req.text, top_n=10),
    }
