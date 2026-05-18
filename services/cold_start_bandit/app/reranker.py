"""Maximal Marginal Relevance (MMR) diversity rerank.

After LinUCB gives us a scored shortlist, we rerank to break filter bubbles:

    MMR(i) = lambda * relevance(i) - (1 - lambda) * max_{j in selected} sim(i, j)

where ``sim`` is cosine over content embeddings. lambda=1 is pure relevance
(what YouTube optimizes); lambda~0.7 visibly diversifies while keeping
relevance dominant.

This is the same family as Pareto multi-objective ranking, kept simple: we
fold creator-equity in as an optional penalty on per-creator over-representation
in the selected set. That avoids the "30% of the feed is one creator" failure
mode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CandidateForRerank:
    video_id: str
    creator_id: str
    relevance: float            # the LinUCB total score
    embedding: np.ndarray       # content embedding, L2-normalized recommended


def mmr_rerank(
    candidates: list[CandidateForRerank],
    top_k: int,
    lambda_: float = 0.7,
    creator_penalty: float = 0.1,
) -> list[int]:
    """Return indices into ``candidates`` in MMR-selected order, len <= top_k.

    ``creator_penalty`` is a per-additional-pick penalty for repeating a
    creator already in the selected set. Default 0.1 ≈ "second video by the
    same creator is fine; third starts to lose."
    """
    if not candidates:
        return []
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError("lambda must be in [0, 1]")
    n = len(candidates)
    if top_k >= n:
        # Still apply diversity-aware ordering, just over the full set.
        top_k = n

    # Normalize embeddings for cosine.
    embs = np.stack([_norm(c.embedding) for c in candidates])
    relevances = np.asarray([c.relevance for c in candidates], dtype=np.float64)
    creators = [c.creator_id for c in candidates]

    selected: list[int] = []
    remaining = set(range(n))
    creator_counts: dict[str, int] = {}

    while len(selected) < top_k and remaining:
        best_idx = -1
        best_score = -np.inf
        for i in remaining:
            if not selected:
                penalty = 0.0
            else:
                sims = embs[i] @ embs[selected].T  # (len(selected),)
                penalty = float(np.max(sims))
            creator_pen = creator_penalty * creator_counts.get(creators[i], 0)
            score = lambda_ * relevances[i] - (1.0 - lambda_) * penalty - creator_pen
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
        creator_counts[creators[best_idx]] = creator_counts.get(creators[best_idx], 0) + 1

    return selected


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else v / n
