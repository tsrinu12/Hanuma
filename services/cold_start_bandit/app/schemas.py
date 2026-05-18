"""Request / response models for the bandit service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    """A video the ranker is considering showing to the user."""

    video_id: str
    creator_id: str
    # The V-JEPA / content embedding for the video. The bandit uses this
    # both as part of the context vector and as the dimension MMR diversifies on.
    content_embedding: list[float]
    # Optional: how old (in hours) the video is. Useful as a feature; helps
    # the bandit reason about cold-start vs. established content.
    age_hours: float = Field(default=0.0)
    # Optional: how many impressions the video already has globally. The
    # bandit uses this to widen exploration on truly cold arms.
    impressions_global: int = Field(default=0)


class RankRequest(BaseModel):
    user_id: str
    # User embedding (from V-JEPA over watch history). Same dim as candidate
    # content_embedding.
    user_embedding: list[float]
    candidates: list[Candidate]
    top_k: int = Field(default=10, ge=1, le=200)
    # Optional A/B knob to override exploration. None = use service default.
    alpha: float | None = None
    # Optional override for MMR diversity. None = service default.
    mmr_lambda: float | None = None


class RankedItem(BaseModel):
    video_id: str
    score: float
    # Decomposed UCB score for observability.
    exploit: float
    explore: float
    # Which peer cluster the bandit used to seed this prediction.
    peer_cluster_id: int | None = None
    rank: int


class RankResponse(BaseModel):
    user_id: str
    results: list[RankedItem]
    alpha_used: float
    mmr_lambda_used: float


class RewardEvent(BaseModel):
    """One observed reward — fed back to the bandit to update weights."""

    user_id: str
    video_id: str
    # Context the recommendation was made under. Must match the dim the
    # bandit was configured with.
    context: list[float]
    # Reward signal. Typical mapping:
    #   completion_ratio in [0,1]            -> reward
    #   click but no watch                   -> 0.0
    #   click + watch >=70%                  -> 1.0
    #   shared / liked                       -> 1.2 (clipped at 1.0 if you prefer)
    reward: float


class RewardResponse(BaseModel):
    accepted: bool
    arm_pulls: int


class PeerLookupRequest(BaseModel):
    """Find peer videos for a brand-new upload to seed its initial routing."""

    content_embedding: list[float]
    k: int = Field(default=20, ge=1, le=200)


class PeerLookupResponse(BaseModel):
    neighbors: list[str]
    distances: list[float]
    cluster_id: int


class IndexVideoRequest(BaseModel):
    """Index a video into the peer-cluster store. Called by the ingest pipeline."""

    video_id: str
    content_embedding: list[float]


class IndexVideoResponse(BaseModel):
    indexed: bool
    cluster_id: int


HealthStatus = Literal["ok", "degraded", "starting"]
