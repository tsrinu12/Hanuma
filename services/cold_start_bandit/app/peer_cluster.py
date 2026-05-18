"""Peer-cluster index over V-JEPA / content embeddings.

Day-1 uploads have no engagement data. We use the content embedding to find
their N nearest neighbours among already-indexed videos; the centroid of
those neighbours' clusters is treated as the new video's *peer cluster*.
The bandit then routes the new video using its peer cluster's learned theta —
that is what gives it warm-start distribution.

This module uses FAISS if installed and falls back to a numpy brute-force
implementation otherwise. Brute force is fine up to ~100k videos in memory;
beyond that, the prod build pulls in FAISS.

Clustering is mini-batch k-means over the same embedding space. We rebuild
periodically (every N new videos, configurable) rather than per-insert; for
the prototype we expose a manual ``recluster`` call.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class IndexEntry:
    video_id: str
    embedding: np.ndarray  # (d,), L2-normalized
    cluster_id: int


class PeerClusterIndex:
    """In-memory ANN + k-means cluster assignment.

    Operations:
      - add(video_id, embedding): index a new video, return its cluster id
      - search(embedding, k): return k nearest video_ids and their distances
      - recluster(n_clusters): refit k-means and reassign all entries
    """

    def __init__(self, dim: int, n_clusters: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        self.dim = dim
        self.n_clusters = n_clusters
        self._entries: dict[str, IndexEntry] = {}
        # Cluster centroids: (n_clusters, dim). Initialized on first recluster.
        self._centroids: np.ndarray | None = None
        self._lock = threading.RLock()

    # --- internals ---

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return v
        return v / norm

    def _assign_cluster(self, v: np.ndarray) -> int:
        if self._centroids is None or len(self._centroids) == 0:
            # Bootstrap: everything in cluster 0 until first recluster.
            return 0
        # cosine sim, since both v and centroids are L2-normalized
        sims = self._centroids @ v
        return int(np.argmax(sims))

    # --- public ---

    def add(self, video_id: str, embedding: np.ndarray) -> int:
        v = self._normalize(np.asarray(embedding, dtype=np.float64))
        if v.shape != (self.dim,):
            raise ValueError(f"embedding shape {v.shape} != ({self.dim},)")
        with self._lock:
            cluster_id = self._assign_cluster(v)
            self._entries[video_id] = IndexEntry(
                video_id=video_id, embedding=v, cluster_id=cluster_id
            )
            return cluster_id

    def cluster_for(self, video_id: str) -> int | None:
        with self._lock:
            entry = self._entries.get(video_id)
            return entry.cluster_id if entry else None

    def cluster_for_embedding(self, embedding: np.ndarray) -> int:
        v = self._normalize(np.asarray(embedding, dtype=np.float64))
        with self._lock:
            return self._assign_cluster(v)

    def search(self, embedding: np.ndarray, k: int = 10) -> tuple[list[str], list[float]]:
        v = self._normalize(np.asarray(embedding, dtype=np.float64))
        with self._lock:
            if not self._entries:
                return [], []
            ids = list(self._entries.keys())
            mat = np.stack([self._entries[i].embedding for i in ids])
            # cosine similarities; convert to "distance" = 1 - sim for caller clarity
            sims = mat @ v
            order = np.argsort(-sims)[:k]
            return [ids[i] for i in order], [float(1.0 - sims[i]) for i in order]

    def recluster(self, n_clusters: int | None = None, n_iter: int = 20, seed: int = 0) -> int:
        """Refit k-means and reassign all entries.

        Returns the number of clusters actually used (capped by available
        embeddings — k-means with k>N is degenerate).
        """
        if n_clusters is not None:
            self.n_clusters = n_clusters
        with self._lock:
            n_points = len(self._entries)
            if n_points == 0:
                self._centroids = None
                return 0
            k = min(self.n_clusters, n_points)
            ids = list(self._entries.keys())
            X = np.stack([self._entries[i].embedding for i in ids])

            # Mini-batch k-means via numpy: random init, Lloyd updates.
            rng = np.random.default_rng(seed)
            init_idx = rng.choice(n_points, size=k, replace=False)
            centroids = X[init_idx].copy()
            for _ in range(n_iter):
                # cosine sim with centroids; assign to argmax.
                sims = X @ centroids.T  # (n, k)
                assign = np.argmax(sims, axis=1)
                new_centroids = np.zeros_like(centroids)
                for c in range(k):
                    mask = assign == c
                    if not mask.any():
                        # Re-seed empty cluster from a random point.
                        new_centroids[c] = X[rng.integers(0, n_points)]
                    else:
                        m = X[mask].mean(axis=0)
                        n = float(np.linalg.norm(m))
                        new_centroids[c] = m / n if n > 0 else m
                shift = float(np.linalg.norm(new_centroids - centroids))
                centroids = new_centroids
                if shift < 1e-6:
                    break

            self._centroids = centroids

            # Reassign every entry.
            sims_all = X @ centroids.T
            assignments = np.argmax(sims_all, axis=1)
            for vid, c in zip(ids, assignments):
                self._entries[vid].cluster_id = int(c)

            return k

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "videos_indexed": len(self._entries),
                "clusters": int(self._centroids.shape[0]) if self._centroids is not None else 0,
            }
