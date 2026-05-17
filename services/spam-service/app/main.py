"""spam-service: combined ML + rules pipeline for moderation of comments,
captions, and upload metadata.

Endpoints:
  POST /check_text       body: {text, user_id?}             -> {verdict, scores, rules, action}
  POST /check_metadata   body: {title, description, tags?}  -> {verdict, scores, ...}
  POST /check_user       body: {user_id, text?}             -> {rate_limited, repeat_post}

Verdict tiers: clean | review | block
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .classifiers import ClassifierBundle
from .heuristics import rate_limited, repeat_post, rule_scores

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("spam-service")

app = FastAPI(title="distrebute.com - Spam Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_bundle: Optional[ClassifierBundle] = None


def bundle() -> ClassifierBundle:
    global _bundle
    if _bundle is None:
        _bundle = ClassifierBundle()
    return _bundle


# ---- decision policy -----------------------------------------------------


BLOCK_THRESHOLDS = {
    "toxic.toxic": 0.9,
    "toxic.severe_toxic": 0.6,
    "toxic.threat": 0.6,
    "toxic.identity_hate": 0.6,
    "hate.hate_speech": 0.85,
    "spam_intent.scam or phishing": 0.85,
}
REVIEW_THRESHOLDS = {
    "toxic.toxic": 0.6,
    "toxic.insult": 0.6,
    "hate.hate_speech": 0.5,
    "spam_intent.spam": 0.6,
    "spam_intent.advertisement": 0.75,
}


def _flat(scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {f"{k}.{kk}": vv for k, sub in scores.items() for kk, vv in sub.items()}


def _decide(flat: Dict[str, float], rule_flags: Dict[str, float]) -> str:
    for k, t in BLOCK_THRESHOLDS.items():
        if flat.get(k, 0.0) >= t:
            return "block"
    if "too_many_urls" in rule_flags and rule_flags["too_many_urls"] >= 4:
        return "block"
    for k, t in REVIEW_THRESHOLDS.items():
        if flat.get(k, 0.0) >= t:
            return "review"
    if rule_flags:
        return "review"
    return "clean"


# ---- schemas -------------------------------------------------------------


class CheckTextReq(BaseModel):
    text: str
    user_id: Optional[str] = None


class CheckMetadataReq(BaseModel):
    title: str
    description: str = ""
    tags: List[str] = []


class CheckUserReq(BaseModel):
    user_id: str
    text: Optional[str] = None


# ---- routes --------------------------------------------------------------


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/check_text")
def check_text(req: CheckTextReq):
    rule_flags = rule_scores(req.text)
    user_signals = {}
    if req.user_id:
        user_signals["rate_limited"] = rate_limited(req.user_id)
        user_signals["repeat_post"] = repeat_post(req.user_id, req.text)
        if user_signals["rate_limited"] or user_signals["repeat_post"]:
            rule_flags["user_abuse"] = 1.0

    scores = bundle().score(req.text)
    flat = _flat(scores)
    verdict = _decide(flat, rule_flags)
    return {
        "verdict": verdict,
        "scores": scores,
        "rules": rule_flags,
        "user": user_signals,
        "action": {"clean": "publish", "review": "queue_for_moderator", "block": "reject"}[verdict],
    }


@app.post("/check_metadata")
def check_metadata(req: CheckMetadataReq):
    combined = f"{req.title}\n{req.description}\n{' '.join(req.tags)}"
    return check_text(CheckTextReq(text=combined))


@app.post("/check_user")
def check_user(req: CheckUserReq):
    return {
        "user_id": req.user_id,
        "rate_limited": rate_limited(req.user_id),
        "repeat_post": repeat_post(req.user_id, req.text or ""),
    }
