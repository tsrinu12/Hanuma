"""Tests for the late-interaction retrieval index."""

import numpy as np

from app.embedder import HashEmbedder
from app.retrieval import ChapterChunk, LateInteractionIndex


def _make_chunk(
    embedder: HashEmbedder, video_id: str, chapter_id: int, text: str
) -> ChapterChunk:
    pooled = embedder.embed_pooled([text])[0]
    tokens = embedder.embed_tokens(text, max_tokens=32)
    return ChapterChunk(
        video_id=video_id,
        chapter_id=chapter_id,
        start_s=0.0,
        end_s=10.0,
        title=text[:40],
        snippet=text,
        pooled=pooled,
        tokens=tokens,
    )


def test_index_grows_with_adds():
    e = HashEmbedder(dim=64)
    idx = LateInteractionIndex()
    idx.add(_make_chunk(e, "v1", 0, "alpha beta gamma"))
    idx.add(_make_chunk(e, "v1", 1, "delta epsilon"))
    idx.add(_make_chunk(e, "v2", 0, "zeta eta theta"))
    assert idx.size() == 3
    assert idx.video_count() == 2


def test_search_prefers_lexical_overlap():
    e = HashEmbedder(dim=128)
    idx = LateInteractionIndex()
    idx.add(_make_chunk(e, "v1", 0, "discussion about pricing strategy and tiers"))
    idx.add(_make_chunk(e, "v1", 1, "we talk football scores all afternoon"))
    idx.add(_make_chunk(e, "v2", 0, "soccer is awesome and great"))

    q = "pricing strategy"
    pooled = e.embed_pooled([q])[0]
    tokens = e.embed_tokens(q, max_tokens=8)
    hits = idx.search(pooled, tokens, top_k=2)
    assert hits[0][0].video_id == "v1"
    assert hits[0][0].chapter_id == 0


def test_search_video_filter():
    e = HashEmbedder(dim=64)
    idx = LateInteractionIndex()
    idx.add(_make_chunk(e, "v1", 0, "pricing strategy"))
    idx.add(_make_chunk(e, "v2", 0, "pricing strategy"))
    q = "pricing"
    pooled = e.embed_pooled([q])[0]
    tokens = e.embed_tokens(q, max_tokens=4)
    hits = idx.search(pooled, tokens, top_k=5, video_id="v1")
    assert all(h[0].video_id == "v1" for h in hits)


def test_empty_search():
    e = HashEmbedder(dim=64)
    idx = LateInteractionIndex()
    pooled = e.embed_pooled(["hi"])[0]
    tokens = e.embed_tokens("hi", max_tokens=2)
    assert idx.search(pooled, tokens, top_k=5) == []
