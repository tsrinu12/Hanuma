"""AI service - exposes /transcribe, /summarize, /moderate AND runs a background queue worker.

The background worker consumes `ai_jobs` from Redis (pushed by video-service),
downloads the raw video from S3, runs the full AI pipeline, and writes results to
metadata-service + search-service.
"""
import json
import logging
import os
import tempfile
import threading
import time

import boto3
import httpx
import redis
from botocore.client import Config
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import settings
from .moderation import moderate_text, moderate_video
from .summarization import highlights_from_segments, summarize
from .transcription import transcribe

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ai")

app = FastAPI(title="distrebute.com - AI Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_r = redis.Redis.from_url(settings.redis_url)


def _s3():
    kwargs = dict(region_name=settings.aws_region, config=Config(signature_version="s3v4"))
    if settings.s3_endpoint_url:
        kwargs["endpoint_url"] = settings.s3_endpoint_url
    if settings.s3_access_key:
        kwargs["aws_access_key_id"] = settings.s3_access_key
        kwargs["aws_secret_access_key"] = settings.s3_secret_key
    return boto3.client("s3", **kwargs)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


class TextReq(BaseModel):
    text: str


@app.post("/summarize")
def summarize_text(req: TextReq):
    return summarize(req.text)


@app.post("/moderate-text")
def moderate_text_ep(req: TextReq):
    return moderate_text(req.text)


class TranscribeReq(BaseModel):
    s3_key: str
    language: str | None = None


@app.post("/transcribe")
def transcribe_ep(req: TranscribeReq):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        path = tmp.name
    try:
        _s3().download_file(settings.s3_bucket_raw, req.s3_key, path)
        return transcribe(path, language=req.language)
    finally:
        try: os.unlink(path)
        except OSError: pass


# ------------------------ Pipeline worker (Redis loop) ------------------------
def run_pipeline(video_id: str, raw_key: str) -> None:
    log.info("AI pipeline start: %s", video_id)
    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "v.mp4")
        _s3().download_file(settings.s3_bucket_raw, raw_key, local)

        # 1) Moderation (visual)
        mod = moderate_video(local)
        if mod["blocked"]:
            log.warning("Video %s blocked: nsfw=%.2f", video_id, mod["nsfw_max"])
            httpx.patch(f"{settings.metadata_url}/videos/{video_id}",
                        json={"status": "blocked"}, timeout=15.0)
            httpx.post(f"{settings.metadata_url}/ai-outputs",
                       json={"video_id": video_id, "kind": "moderation", "payload": mod},
                       timeout=15.0)
            return

        # 2) Transcription
        tr = transcribe(local)
        text_mod = moderate_text(tr["full_text"])
        if text_mod["text_blocked"]:
            mod["text_blocked"] = True
            mod["text_hits"] = text_mod["hits"]
            httpx.patch(f"{settings.metadata_url}/videos/{video_id}",
                        json={"status": "blocked"}, timeout=15.0)
            httpx.post(f"{settings.metadata_url}/ai-outputs",
                       json={"video_id": video_id, "kind": "moderation", "payload": mod},
                       timeout=15.0)
            return

        httpx.post(f"{settings.metadata_url}/transcripts",
                   json={"video_id": video_id, "language": tr["language"],
                         "full_text": tr["full_text"], "segments": tr["segments"]},
                   timeout=60.0)

        # 3) Summarisation + highlights
        summ = summarize(tr["full_text"])
        hl = highlights_from_segments(tr["segments"])
        httpx.post(f"{settings.metadata_url}/ai-outputs",
                   json={"video_id": video_id, "kind": "summary", "payload": summ},
                   timeout=30.0)
        httpx.post(f"{settings.metadata_url}/ai-outputs",
                   json={"video_id": video_id, "kind": "highlights", "payload": hl},
                   timeout=30.0)
        httpx.post(f"{settings.metadata_url}/ai-outputs",
                   json={"video_id": video_id, "kind": "moderation", "payload": mod},
                   timeout=30.0)

        # 4) Embed transcript chunks into search-service
        chunks = []
        cur, cur_start = "", None
        for seg in tr["segments"]:
            if cur_start is None:
                cur_start = seg["start"]
            cur += " " + seg["text"]
            if len(cur) > 500:
                chunks.append({"start": cur_start, "end": seg["end"], "text": cur.strip()})
                cur, cur_start = "", None
        if cur.strip():
            chunks.append({"start": cur_start or 0.0,
                           "end": tr["segments"][-1]["end"] if tr["segments"] else 0,
                           "text": cur.strip()})
        try:
            httpx.post(f"{settings.search_url}/index",
                       json={"video_id": video_id, "chunks": chunks},
                       timeout=120.0)
        except Exception as e:
            log.warning("search indexing failed: %s", e)

    log.info("AI pipeline done: %s", video_id)


def _worker_loop():
    log.info("AI worker started, listening on redis list 'ai_jobs'")
    while True:
        try:
            item = _r.brpop("ai_jobs", timeout=5)
            if not item:
                continue
            _, payload = item
            payload = payload.decode() if isinstance(payload, bytes) else payload
            video_id, raw_key = payload.split("|", 1)
            try:
                run_pipeline(video_id, raw_key)
            except Exception as e:
                log.exception("pipeline failed: %s", e)
        except Exception as e:
            log.exception("worker loop error: %s", e)
            time.sleep(2)


@app.on_event("startup")
def _start_worker():
    if os.getenv("AI_WORKER", "1") == "1":
        threading.Thread(target=_worker_loop, daemon=True).start()


class EnqueueReq(BaseModel):
    video_id: str
    raw_s3_key: str


@app.post("/enqueue")
def enqueue(req: EnqueueReq):
    _r.lpush("ai_jobs", f"{req.video_id}|{req.raw_s3_key}")
    return {"ok": True}
