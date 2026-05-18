"""FastAPI entrypoint for the semantic search service.

Endpoints:
    GET  /health
    POST /index_video         -> ingest a video: detect/accept shots, assemble
                                 chapters, embed, and index for retrieval
    POST /search              -> ColBERT-style search; optional video_id filter
    GET  /admin/snapshot      -> index size, configured profile
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI

from shared.logging import configure_logging, log_event
from shared.schemas import HealthResponse

from .chapter_pipeline import (
    assemble_chapters,
    synthesize_shots_from_transcript,
    to_api_chapters,
)
from .config import Settings, get_settings
from .embedder import build_embedder
from .retrieval import ChapterChunk, LateInteractionIndex
from .schemas import (
    IndexVideoRequest,
    IndexVideoResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(service=settings.service_name, level=settings.log_level)
    app.state.settings = settings
    app.state.embedder = build_embedder(settings)
    app.state.index = LateInteractionIndex()
    log_event(
        logger,
        "service_starting",
        service=settings.service_name,
        profile=settings.model_profile.value,
        token_embed_dim=settings.token_embed_dim,
    )
    yield
    log_event(logger, "service_stopping")


app = FastAPI(title="distrebute semantic search", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s: Settings = app.state.settings
    return HealthResponse(service=s.service_name, model_profile=s.model_profile.value)


@app.post("/index_video", response_model=IndexVideoResponse)
def index_video(req: IndexVideoRequest) -> IndexVideoResponse:
    settings: Settings = app.state.settings
    embedder = app.state.embedder
    index: LateInteractionIndex = app.state.index

    shots = req.shots or synthesize_shots_from_transcript(req.transcript)

    shot_embs: np.ndarray | None = None
    if req.shot_embeddings is not None:
        shot_embs = np.asarray(req.shot_embeddings, dtype=np.float64)

    assembled = assemble_chapters(
        shots=shots,
        transcript=req.transcript,
        shot_embeddings=shot_embs,
        merge_distance=settings.chapter_merge_distance,
        max_chapter_seconds=settings.max_chapter_seconds,
    )

    # Embed and index every chapter.
    pooled = embedder.embed_pooled([c.text or c.title for c in assembled])
    chunks: list[ChapterChunk] = []
    for i, c in enumerate(assembled):
        tokens = embedder.embed_tokens(
            c.text or c.title, max_tokens=settings.max_tokens_per_chapter
        )
        snippet = _make_snippet(c.text)
        chunks.append(
            ChapterChunk(
                video_id=req.video_id,
                chapter_id=c.chapter_id,
                start_s=c.start_s,
                end_s=c.end_s,
                title=c.title,
                snippet=snippet,
                pooled=pooled[i],
                tokens=tokens,
            )
        )
    n = index.add_many(chunks)
    log_event(
        logger,
        "video_indexed",
        video_id=req.video_id,
        chapters=len(assembled),
        chunks=n,
    )
    return IndexVideoResponse(
        video_id=req.video_id,
        chapters=to_api_chapters(assembled),
        chunks_indexed=n,
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    settings: Settings = app.state.settings
    embedder = app.state.embedder
    index: LateInteractionIndex = app.state.index

    if index.size() == 0:
        return SearchResponse(query=req.query, hits=[])

    pooled = embedder.embed_pooled([req.query])[0]
    tokens = embedder.embed_tokens(req.query, max_tokens=settings.max_tokens_per_chapter)
    results = index.search(
        query_pooled=pooled,
        query_tokens=tokens,
        top_k=req.top_k,
        video_id=req.video_id,
    )
    hits = [
        SearchHit(
            video_id=c.video_id,
            chapter_id=c.chapter_id,
            start_s=c.start_s,
            end_s=c.end_s,
            title=c.title,
            score=score,
            snippet=c.snippet,
        )
        for c, score in results
    ]
    log_event(
        logger,
        "search_served",
        query=req.query,
        hits=len(hits),
        video_id=req.video_id,
    )
    return SearchResponse(query=req.query, hits=hits)


@app.get("/admin/snapshot")
def admin_snapshot() -> dict:
    settings: Settings = app.state.settings
    index: LateInteractionIndex = app.state.index
    return {"profile": settings.model_profile.value, **index.stats()}


def _make_snippet(text: str, max_chars: int = 220) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."
