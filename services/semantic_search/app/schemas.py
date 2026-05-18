"""Pydantic models for the semantic search service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """A single ASR/subtitle line with timestamps."""

    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    text: str


class Shot(BaseModel):
    """A continuous camera shot."""

    shot_id: int
    start_s: float
    end_s: float


class Chapter(BaseModel):
    """One semantically coherent span — multiple shots merged together."""

    chapter_id: int
    start_s: float
    end_s: float
    title: str
    summary: str | None = None
    # Indices of shots that compose this chapter (for debugging).
    shot_ids: list[int]


class IndexVideoRequest(BaseModel):
    video_id: str
    duration_s: float
    transcript: list[TranscriptSegment]
    # Either raw shot boundaries from your pipeline, or omit and we'll
    # synthesize from transcript pauses + frame embeddings (lite mode).
    shots: list[Shot] | None = None
    # Optional: precomputed per-shot keyframe embeddings (e.g. CLIP). Same
    # length as `shots`. We use them for chapter merging when supplied.
    shot_embeddings: list[list[float]] | None = None


class IndexVideoResponse(BaseModel):
    video_id: str
    chapters: list[Chapter]
    chunks_indexed: int


class SearchRequest(BaseModel):
    query: str
    # Filter to a single video for "find the part where..." search; omit for
    # cross-video search.
    video_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=50)


class SearchHit(BaseModel):
    video_id: str
    chapter_id: int
    start_s: float
    end_s: float
    title: str
    score: float
    # Most-matched sentence in the chapter, for highlight rendering.
    snippet: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
