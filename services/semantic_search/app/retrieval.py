"""Late-interaction retrieval (ColBERT MaxSim) over per-chapter token embeddings.

Each indexed chapter stores ``(n_tokens, d)`` token embeddings. For a query
with ``(m_tokens, d)`` token embeddings:

    score(chapter | query) = sum_q max_t  q_emb[q] . chapter_token_emb[t]

This is the "MaxSim" interaction from ColBERT. It's strictly more powerful
than dot-product over pooled vectors at the cost of O(m * n) per chapter —
manageable because token counts are clipped to ``max_tokens_per_chapter``.

We use a two-stage retrieval:

  1. Coarse pool prefilter: pooled-embedding dot product over all chapters
     (very fast, narrows to ~50 chapters).
  2. Re-rank the prefilter top-N with MaxSim.

Storage is in-memory dicts. For prod, replace with FAISS pooled-vector
prefilter + on-disk token-vector store (PLAID).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class ChapterChunk:
    video_id: str
    chapter_id: int
    start_s: float
    end_s: float
    title: str
    snippet: str                  # short text used for highlight rendering
    pooled: np.ndarray            # (pooled_dim,) L2-normalized
    tokens: np.ndarray            # (n_tokens, token_dim) L2-normalized rows


class LateInteractionIndex:
    """In-memory ColBERT-ish index."""

    def __init__(self) -> None:
        self._chunks: list[ChapterChunk] = []
        # secondary index by video_id for per-video search
        self._by_video: dict[str, list[int]] = {}
        self._lock = threading.RLock()

    def add(self, chunk: ChapterChunk) -> int:
        with self._lock:
            idx = len(self._chunks)
            self._chunks.append(chunk)
            self._by_video.setdefault(chunk.video_id, []).append(idx)
            return idx

    def add_many(self, chunks: list[ChapterChunk]) -> int:
        n = 0
        for c in chunks:
            self.add(c)
            n += 1
        return n

    def size(self) -> int:
        return len(self._chunks)

    def video_count(self) -> int:
        return len(self._by_video)

    def search(
        self,
        query_pooled: np.ndarray,
        query_tokens: np.ndarray,
        top_k: int,
        prefilter_size: int = 100,
        video_id: str | None = None,
    ) -> list[tuple[ChapterChunk, float]]:
        """Two-stage retrieval. Returns (chunk, score) pairs sorted desc."""
        with self._lock:
            # Candidate set.
            if video_id is not None:
                cand_indices = list(self._by_video.get(video_id, []))
            else:
                cand_indices = list(range(len(self._chunks)))
            if not cand_indices:
                return []

            # Stage 1: pooled-embedding similarity.
            pooled_stack = np.stack([self._chunks[i].pooled for i in cand_indices])
            coarse_scores = pooled_stack @ query_pooled  # (n_cand,)
            n_prefilter = min(prefilter_size, len(cand_indices))
            top_idx_coarse = np.argpartition(-coarse_scores, n_prefilter - 1)[:n_prefilter]
            top_indices = [cand_indices[int(i)] for i in top_idx_coarse]

            # Stage 2: MaxSim re-rank.
            rescored: list[tuple[int, float]] = []
            for ci in top_indices:
                chunk = self._chunks[ci]
                # q_tokens: (m, d); chunk.tokens: (n, d)
                # interaction: (m, n). For each query token, take max over chunk tokens; sum.
                interaction = query_tokens @ chunk.tokens.T
                score = float(np.max(interaction, axis=1).sum())
                rescored.append((ci, score))
            rescored.sort(key=lambda x: -x[1])
            top = rescored[:top_k]
            return [(self._chunks[ci], s) for ci, s in top]

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"chunks": len(self._chunks), "videos": len(self._by_video)}
