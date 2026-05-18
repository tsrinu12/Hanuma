"""Context-vector construction for the bandit.

We can't feed raw V-JEPA embeddings to LinUCB — they're 384-d or 768-d and
LinUCB is per-arm O(d^3) per inversion. Instead we construct a compact
``context_dim``-d feature vector that summarizes the (user, candidate)
interaction:

  - cos similarity between user_embedding and candidate_content_embedding
  - log1p(age_hours) bucketed
  - log1p(impressions_global) bucketed
  - 32-d hashed projection of element-wise (user * candidate) — a low-rank
    interaction signal that LinUCB can pick up

This is intentionally simple and feature-engineered, not learned. The learned
piece is theta_a per peer cluster, which composes well with engineered features.

Extend by adding rows to the returned vector; remember to bump ``context_dim``
in Settings to match.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _hash_project(vec: np.ndarray, out_dim: int, seed: int = 0) -> np.ndarray:
    """Random-feature hashing: deterministic, no allocation of a (d_in, d_out)
    matrix. Maps a d-dim vector to ``out_dim`` via signed hash bucketing.
    """
    out = np.zeros(out_dim, dtype=np.float64)
    for i, v in enumerate(vec):
        h = hashlib.blake2b(f"{seed}:{i}".encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % out_dim
        sign = 1.0 if (h[4] & 1) else -1.0
        out[idx] += sign * float(v)
    return out


def _bucket(value: float, edges: list[float]) -> np.ndarray:
    """One-hot bucket indicator."""
    out = np.zeros(len(edges) + 1, dtype=np.float64)
    for i, e in enumerate(edges):
        if value < e:
            out[i] = 1.0
            return out
    out[-1] = 1.0
    return out


AGE_BUCKETS = [1.0, 6.0, 24.0, 24 * 7.0]  # 0-1h, 1-6h, 6-24h, 1-7d, >7d
IMP_BUCKETS = [10.0, 100.0, 1_000.0, 10_000.0]


def build_context_vector(
    user_embedding: np.ndarray,
    content_embedding: np.ndarray,
    age_hours: float,
    impressions_global: int,
    context_dim: int,
) -> np.ndarray:
    """Return a context_dim-d vector summarising (user, candidate).

    Layout (default context_dim=64):
        [ cos_sim,                                    # 1
          age_bucket_one_hot,                          # 5
          imp_bucket_one_hot,                          # 5
          hashed_interaction_projection,               # 53 ]
    """
    ue = np.asarray(user_embedding, dtype=np.float64)
    ce = np.asarray(content_embedding, dtype=np.float64)
    if ue.shape != ce.shape:
        raise ValueError(
            f"user_embedding shape {ue.shape} != content_embedding shape {ce.shape}"
        )

    # cos similarity
    ue_n = np.linalg.norm(ue) or 1.0
    ce_n = np.linalg.norm(ce) or 1.0
    cos_sim = float((ue @ ce) / (ue_n * ce_n))

    age_oh = _bucket(float(age_hours), AGE_BUCKETS)
    imp_oh = _bucket(float(impressions_global), IMP_BUCKETS)

    head = np.concatenate([[cos_sim], age_oh, imp_oh])  # 1 + 5 + 5 = 11
    remaining = context_dim - head.shape[0]
    if remaining < 0:
        raise ValueError(
            f"context_dim={context_dim} too small; need >= {head.shape[0]}"
        )

    interaction = ue * ce  # element-wise; same dim as embeddings
    hashed = _hash_project(interaction, out_dim=remaining)
    return np.concatenate([head, hashed]).astype(np.float64)
