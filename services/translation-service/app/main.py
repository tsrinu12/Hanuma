"""translation-service: NLLB-200 multilingual translation + fastText language ID.

Endpoints:
  POST /detect          body: {text}                     -> {lang, confidence, candidates}
  POST /translate       body: {text, src_lang, tgt_lang} -> {text, src_lang, tgt_lang}
  POST /translate_batch body: {texts, src_lang, tgt_lang}-> {texts}
  GET  /languages                                        -> supported codes
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .models import ISO_TO_FLORES, LangID, NLLBTranslator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("translation-service")

app = FastAPI(title="distrebute.com - Translation Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_translator: Optional[NLLBTranslator] = None
_langid: Optional[LangID] = None


def translator() -> NLLBTranslator:
    global _translator
    if _translator is None:
        _translator = NLLBTranslator()
    return _translator


def langid() -> LangID:
    global _langid
    if _langid is None:
        _langid = LangID()
    return _langid


class DetectReq(BaseModel):
    text: str
    top_k: int = 3


class TranslateReq(BaseModel):
    text: str
    src_lang: Optional[str] = None  # auto-detect if omitted
    tgt_lang: str


class TranslateBatchReq(BaseModel):
    texts: List[str]
    src_lang: Optional[str] = None
    tgt_lang: str


@app.get("/health")
def health():
    return {"ok": True, "model": settings.nllb_model}


@app.get("/languages")
def languages():
    return {"iso_to_flores": ISO_TO_FLORES, "total_nllb_supported": 200}


@app.post("/detect")
def detect(req: DetectReq):
    cands = langid().detect(req.text, k=req.top_k)
    return {
        "lang": cands[0][0] if cands else "und",
        "confidence": cands[0][1] if cands else 0.0,
        "candidates": [{"lang": c, "confidence": p} for c, p in cands],
    }


@app.post("/translate")
def translate(req: TranslateReq):
    src = req.src_lang or (langid().detect(req.text)[0][0] if req.text else "eng")
    try:
        out = translator().translate([req.text], src_lang=src, tgt_lang=req.tgt_lang)
    except Exception as e:
        raise HTTPException(500, f"translate failed: {e}")
    return {"text": out[0], "src_lang": src, "tgt_lang": req.tgt_lang}


@app.post("/translate_batch")
def translate_batch(req: TranslateBatchReq):
    if not req.texts:
        return {"texts": []}
    src = req.src_lang
    if not src:
        src = langid().detect(req.texts[0])[0][0]
    try:
        outs = translator().translate(req.texts, src_lang=src, tgt_lang=req.tgt_lang)
    except Exception as e:
        raise HTTPException(500, f"translate failed: {e}")
    return {"texts": outs, "src_lang": src, "tgt_lang": req.tgt_lang}
