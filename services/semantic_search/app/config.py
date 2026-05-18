"""Settings for the semantic search service."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field

from shared.config import BaseServiceSettings


class Settings(BaseServiceSettings):
    service_name: str = Field(default="semantic_search")
    port: int = Field(default=8002)

    # --- Embedder ---
    # Lite uses BGE-small (384-d, CPU-friendly). Prod uses BGE-large (1024-d).
    embedder_lite: str = Field(default="BAAI/bge-small-en-v1.5")
    embedder_prod: str = Field(default="BAAI/bge-large-en-v1.5")

    # Token-level (ColBERT-style) embedding dim. We project down from the
    # encoder hidden size to keep memory tractable.
    token_embed_dim: int = Field(default=128)
    # Max tokens to keep per chapter for late-interaction. Truncates long ones.
    max_tokens_per_chapter: int = Field(default=64)

    # --- Shot detection ---
    # Min seconds between shot boundaries (filters flickers).
    min_shot_seconds: float = Field(default=1.5)

    # --- Chapter assembly ---
    # Adjacent shots are merged into chapters when their content embeddings
    # are within this cosine distance.
    chapter_merge_distance: float = Field(default=0.25)
    # Maximum chapter length (seconds).
    max_chapter_seconds: float = Field(default=180.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
