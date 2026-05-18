"""Settings for the cold-start bandit service."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field

from shared.config import BaseServiceSettings


class Settings(BaseServiceSettings):
    service_name: str = Field(default="cold_start_bandit")
    port: int = Field(default=8001)

    # --- LinUCB ---
    # alpha = exploration parameter. Higher = more exploration. 1.0 is a common
    # starting point; tune via simulation.
    bandit_alpha: float = Field(default=1.0)
    # dimensionality of the (user, item) context vector
    context_dim: int = Field(default=64)

    # --- Peer cluster ---
    # dimensionality of V-JEPA / content embeddings used to find look-alikes
    peer_embed_dim: int = Field(default=384)
    peer_k_neighbors: int = Field(default=50)

    # --- MMR rerank ---
    # lambda in MMR: 1.0 = pure relevance, 0.0 = pure diversity.
    mmr_lambda: float = Field(default=0.7)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
