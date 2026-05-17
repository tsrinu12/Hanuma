"""vjepa-service FastAPI app.

Endpoints:
  POST /embed         body: {video_url | s3_key}     -> {embedding: [float], dim}
  POST /classify      body: {video_url | s3_key}     -> {labels: [(name, score)]}
  POST /index         body: {video_id, s3_key}       -> stores embedding in Qdrant
  POST /similar       body: {video_id, top_k}        -> nearest neighbors from Qdrant
"""
from __future__ import annotations

import logging
import os
import tempfile
import uuid
from typing import List, Optional

import boto3
import httpx
from botocore.client import Config
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from .config import settings
from .vjepa import VJEPAEngine

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vjepa-service")

app = FastAPI(title="distrebute.com - V-JEPA Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Lazy-load on first request so the service can boot quickly in k8s
_engine: Optional[VJEPAEngine] = None


def engine() -> VJEPAEngine:
    global _engine
    if _engine is None:
        _engine = VJEPAEngine()
    return _engine


_qdrant = QdrantClient(url=settings.qdrant_url)


def _ensure_collection() -> None:
    cols = {c.name for c in _qdrant.get_collections().collections}
    if settings.qdrant_collection not in cols:
        _qdrant.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=settings.embed_dim, distance=Distance.COSINE),
        )


def _s3():
    kwargs = dict(region_name=settings.aws_region, config=Config(signature_version="s3v4"))
    if settings.s3_endpoint_url:
        kwargs["endpoint_url"] = settings.s3_endpoint_url
    if settings.s3_access_key:
        kwargs["aws_access_key_id"] = settings.s3_access_key
        kwargs["aws_secret_access_key"] = settings.s3_secret_key
    return boto3.client("s3", **kwargs)


def _fetch_video(s3_key: Optional[str], video_url: Optional[str]) -> str:
    """Download to a temp file and return path."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    if s3_key:
        _s3().download_file(settings.s3_bucket_raw, s3_key, path)
    elif video_url:
        with httpx.stream("GET", video_url, timeout=60.0) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
    else:
        raise HTTPException(400, "Provide either s3_key or video_url")
    return path


# ---- schemas --------------------------------------------------------------


class VideoRef(BaseModel):
    s3_key: Optional[str] = None
    video_url: Optional[str] = None


class IndexReq(BaseModel):
    video_id: str
    s3_key: Optional[str] = None
    video_url: Optional[str] = None


class SimilarReq(BaseModel):
    video_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    top_k: int = 10


# ---- routes ---------------------------------------------------------------


@app.get("/health")
def health():
    return {"ok": True, "model": settings.vjepa_model}


@app.post("/embed")
def embed(req: VideoRef):
    path = _fetch_video(req.s3_key, req.video_url)
    try:
        vec = engine().embed(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return {"embedding": vec, "dim": len(vec)}


@app.post("/classify")
def classify(req: VideoRef):
    path = _fetch_video(req.s3_key, req.video_url)
    try:
        labels = engine().classify(path, top_k=5)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return {"labels": [{"name": n, "score": s} for n, s in labels]}


@app.post("/index")
def index(req: IndexReq):
    _ensure_collection()
    path = _fetch_video(req.s3_key, req.video_url)
    try:
        vec = engine().embed(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, req.video_id))
    _qdrant.upsert(
        collection_name=settings.qdrant_collection,
        points=[PointStruct(id=point_id, vector=vec, payload={"video_id": req.video_id})],
    )
    return {"indexed": req.video_id, "dim": len(vec)}


@app.post("/similar")
def similar(req: SimilarReq):
    _ensure_collection()
    if req.embedding is not None:
        query = req.embedding
    elif req.video_id is not None:
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, req.video_id))
        rec = _qdrant.retrieve(collection_name=settings.qdrant_collection, ids=[pid], with_vectors=True)
        if not rec:
            raise HTTPException(404, "video_id not indexed")
        query = rec[0].vector
    else:
        raise HTTPException(400, "Provide video_id or embedding")

    hits = _qdrant.search(
        collection_name=settings.qdrant_collection,
        query_vector=query,
        limit=req.top_k,
    )
    return {
        "results": [
            {"video_id": h.payload.get("video_id"), "score": float(h.score)} for h in hits
        ]
    }
