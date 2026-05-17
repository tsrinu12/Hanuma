"""Search service - semantic search over transcript chunks.

Indexes transcript chunks { video_id, start, end, text } into Qdrant
keyed by sentence-transformer embeddings, and serves a /search endpoint.
"""
import logging
import uuid
from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sentence_transformers import SentenceTransformer

from .config import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("search")

app = FastAPI(title="distrebute.com - Search Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    log.info("Loading embedding model %s", settings.embedding_model)
    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)


@app.on_event("startup")
def _ensure_collection():
    dim = _model().get_sentence_embedding_dimension()
    cols = [c.name for c in _qdrant().get_collections().collections]
    if settings.collection not in cols:
        _qdrant().create_collection(
            collection_name=settings.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        log.info("Created Qdrant collection %s (dim=%s)", settings.collection, dim)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


class Chunk(BaseModel):
    start: float
    end: float
    text: str


class IndexReq(BaseModel):
    video_id: str
    chunks: List[Chunk]


@app.post("/index")
def index(req: IndexReq):
    if not req.chunks:
        return {"indexed": 0}
    texts = [c.text for c in req.chunks]
    vectors = _model().encode(texts, batch_size=32, show_progress_bar=False).tolist()
    points = [
        qm.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "video_id": req.video_id,
                "start": c.start, "end": c.end, "text": c.text,
            },
        )
        for c, vec in zip(req.chunks, vectors)
    ]
    _qdrant().upsert(collection_name=settings.collection, points=points)
    return {"indexed": len(points)}


class SearchReq(BaseModel):
    q: str
    top_k: int = 10
    video_id: str | None = None


@app.post("/search")
def search(req: SearchReq):
    if not req.q.strip():
        raise HTTPException(400, "empty query")
    vec = _model().encode([req.q])[0].tolist()
    flt = None
    if req.video_id:
        flt = qm.Filter(must=[qm.FieldCondition(
            key="video_id", match=qm.MatchValue(value=req.video_id))])
    hits = _qdrant().search(
        collection_name=settings.collection,
        query_vector=vec, limit=req.top_k, query_filter=flt,
    )
    return [
        {"score": h.score, **h.payload}
        for h in hits
    ]


@app.delete("/index/{video_id}")
def remove_video(video_id: str):
    _qdrant().delete(
        collection_name=settings.collection,
        points_selector=qm.FilterSelector(filter=qm.Filter(
            must=[qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id))]
        )),
    )
    return {"deleted": True}
