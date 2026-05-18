"""FastAPI entrypoint for the cold-start bandit service.

Endpoints:
    GET  /health                  -> service health
    POST /index_video             -> index a new upload's content embedding
    POST /peer_lookup             -> find peer videos for an embedding
    POST /rank                    -> rank candidates for a user (LinUCB + MMR)
    POST /reward                  -> feed back observed reward
    POST /admin/recluster         -> rebuild k-means clusters
    GET  /admin/snapshot          -> bandit + index diagnostics
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from shared.logging import configure_logging, log_event
from shared.schemas import HealthResponse

from .bandit import LinUCB
from .config import Settings, get_settings
from .features import build_context_vector
from .peer_cluster import PeerClusterIndex
from .reranker import CandidateForRerank, mmr_rerank
from .schemas import (
    IndexVideoRequest,
    IndexVideoResponse,
    PeerLookupRequest,
    PeerLookupResponse,
    RankedItem,
    RankRequest,
    RankResponse,
    RewardEvent,
    RewardResponse,
)


logger = logging.getLogger(__name__)


def build_state(settings: Settings) -> dict:
    return {
        "bandit": LinUCB(context_dim=settings.context_dim, alpha=settings.bandit_alpha),
        "index": PeerClusterIndex(dim=settings.peer_embed_dim, n_clusters=64),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(service=settings.service_name, level=settings.log_level)
    app.state.settings = settings
    app.state.components = build_state(settings)
    log_event(
        logger,
        "service_starting",
        service=settings.service_name,
        profile=settings.model_profile.value,
        context_dim=settings.context_dim,
        peer_embed_dim=settings.peer_embed_dim,
    )
    yield
    log_event(logger, "service_stopping")


app = FastAPI(title="distrebute cold-start bandit", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings: Settings = app.state.settings
    return HealthResponse(
        service=settings.service_name, model_profile=settings.model_profile.value
    )


@app.post("/index_video", response_model=IndexVideoResponse)
def index_video(req: IndexVideoRequest) -> IndexVideoResponse:
    components = app.state.components
    idx: PeerClusterIndex = components["index"]
    settings: Settings = app.state.settings
    emb = np.asarray(req.content_embedding, dtype=np.float64)
    if emb.shape != (settings.peer_embed_dim,):
        raise HTTPException(
            400,
            detail=f"content_embedding must be {settings.peer_embed_dim}-d, got {emb.shape[0]}",
        )
    cluster_id = idx.add(req.video_id, emb)
    log_event(logger, "video_indexed", video_id=req.video_id, cluster_id=cluster_id)
    return IndexVideoResponse(indexed=True, cluster_id=cluster_id)


@app.post("/peer_lookup", response_model=PeerLookupResponse)
def peer_lookup(req: PeerLookupRequest) -> PeerLookupResponse:
    components = app.state.components
    idx: PeerClusterIndex = components["index"]
    settings: Settings = app.state.settings
    emb = np.asarray(req.content_embedding, dtype=np.float64)
    if emb.shape != (settings.peer_embed_dim,):
        raise HTTPException(400, detail=f"embedding must be {settings.peer_embed_dim}-d")
    neighbors, distances = idx.search(emb, k=req.k)
    cluster_id = idx.cluster_for_embedding(emb)
    return PeerLookupResponse(neighbors=neighbors, distances=distances, cluster_id=cluster_id)


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest) -> RankResponse:
    components = app.state.components
    bandit: LinUCB = components["bandit"]
    idx: PeerClusterIndex = components["index"]
    settings: Settings = app.state.settings

    alpha = req.alpha if req.alpha is not None else settings.bandit_alpha
    lam = req.mmr_lambda if req.mmr_lambda is not None else settings.mmr_lambda

    user_emb = np.asarray(req.user_embedding, dtype=np.float64)
    if user_emb.shape != (settings.peer_embed_dim,):
        raise HTTPException(
            400, detail=f"user_embedding must be {settings.peer_embed_dim}-d"
        )

    if not req.candidates:
        return RankResponse(
            user_id=req.user_id, results=[], alpha_used=alpha, mmr_lambda_used=lam
        )

    # Score every candidate via LinUCB.
    scored: list[tuple[str, str, float, float, float, int, np.ndarray]] = []
    for c in req.candidates:
        ce = np.asarray(c.content_embedding, dtype=np.float64)
        if ce.shape != (settings.peer_embed_dim,):
            raise HTTPException(
                400, detail=f"candidate {c.video_id} embedding wrong dim"
            )
        ctx = build_context_vector(
            user_embedding=user_emb,
            content_embedding=ce,
            age_hours=c.age_hours,
            impressions_global=c.impressions_global,
            context_dim=settings.context_dim,
        )
        cluster_id = idx.cluster_for(c.video_id)
        if cluster_id is None:
            cluster_id = idx.cluster_for_embedding(ce)
        score, exploit, explore = bandit.score(cluster_id, ctx, alpha=alpha)
        scored.append((c.video_id, c.creator_id, score, exploit, explore, cluster_id, ce))

    # Diversity rerank.
    rerank_inputs = [
        CandidateForRerank(
            video_id=s[0], creator_id=s[1], relevance=s[2], embedding=s[6]
        )
        for s in scored
    ]
    order = mmr_rerank(rerank_inputs, top_k=req.top_k, lambda_=lam)

    # Build response.
    results: list[RankedItem] = []
    for rank_i, idx_in_input in enumerate(order):
        s = scored[idx_in_input]
        results.append(
            RankedItem(
                video_id=s[0],
                score=s[2],
                exploit=s[3],
                explore=s[4],
                peer_cluster_id=s[5],
                rank=rank_i,
            )
        )

    log_event(
        logger,
        "rank_served",
        user_id=req.user_id,
        candidates=len(req.candidates),
        top_k=len(results),
        alpha=alpha,
        mmr_lambda=lam,
    )
    return RankResponse(
        user_id=req.user_id, results=results, alpha_used=alpha, mmr_lambda_used=lam
    )


@app.post("/reward", response_model=RewardResponse)
def reward(req: RewardEvent) -> RewardResponse:
    components = app.state.components
    bandit: LinUCB = components["bandit"]
    idx: PeerClusterIndex = components["index"]
    settings: Settings = app.state.settings

    ctx = np.asarray(req.context, dtype=np.float64)
    if ctx.shape != (settings.context_dim,):
        raise HTTPException(400, detail=f"context must be {settings.context_dim}-d")
    cluster_id = idx.cluster_for(req.video_id)
    if cluster_id is None:
        raise HTTPException(404, detail=f"video_id {req.video_id} not indexed")

    arm = bandit.update(cluster_id, ctx, req.reward)
    log_event(
        logger,
        "reward_observed",
        video_id=req.video_id,
        cluster_id=cluster_id,
        reward=req.reward,
        arm_pulls=arm.pulls,
    )
    return RewardResponse(accepted=True, arm_pulls=arm.pulls)


@app.post("/admin/recluster")
def admin_recluster(n_clusters: int = 64) -> dict:
    components = app.state.components
    idx: PeerClusterIndex = components["index"]
    k = idx.recluster(n_clusters=n_clusters)
    log_event(logger, "recluster_done", clusters_used=k)
    return {"clusters_used": k, **idx.stats()}


@app.get("/admin/snapshot")
def admin_snapshot() -> dict:
    components = app.state.components
    bandit: LinUCB = components["bandit"]
    idx: PeerClusterIndex = components["index"]
    return {
        "bandit_arms": bandit.arm_count(),
        "bandit_pulls": bandit.total_pulls(),
        "per_arm": bandit.snapshot(),
        "index": idx.stats(),
    }
