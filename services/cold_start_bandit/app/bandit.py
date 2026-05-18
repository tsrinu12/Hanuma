"""LinUCB contextual bandit (Li et al., 2010).

For each arm a (= a peer cluster, in our setup), we maintain:
    A_a = I_d + sum_t x_t x_t^T          # (d x d)  ridge-regularised design
    b_a = sum_t r_t x_t                  # (d,)     reward-weighted sum
    theta_a = A_a^-1 b_a                 # (d,)     learned weights

Given context x at time t:
    score(a, x) = theta_a . x + alpha * sqrt(x^T A_a^-1 x)
                  ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^
                  exploit         explore (upper confidence width)

Why arms = peer clusters and not individual videos?
    There are millions of videos; per-video LinUCB never accumulates enough
    pulls per arm to converge. Peer clusters (~1k-10k of them, learned over
    V-JEPA embeddings) give us arms that *generalise* — a new video inherits
    its cluster's learned theta on day one. That is the cold-start fix.

The arm chosen for a candidate video is determined by its peer cluster id;
exploitation/exploration is computed against that cluster's (A, b).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArmStats:
    """Per-arm LinUCB state. Maintained behind a lock for thread safety."""

    A: np.ndarray  # (d, d)
    b: np.ndarray  # (d,)
    pulls: int = 0
    rewards_sum: float = 0.0
    # Lazily computed and invalidated on update.
    _A_inv: np.ndarray | None = field(default=None, repr=False)
    _theta: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def initialize(cls, dim: int) -> "ArmStats":
        return cls(A=np.eye(dim, dtype=np.float64), b=np.zeros(dim, dtype=np.float64))

    def A_inv(self) -> np.ndarray:
        if self._A_inv is None:
            # Inversion is O(d^3) but d is small (=context_dim, default 64)
            # and we cache. For d > ~256 switch to Sherman-Morrison rank-1
            # updates per observation.
            self._A_inv = np.linalg.inv(self.A)
        return self._A_inv

    def theta(self) -> np.ndarray:
        if self._theta is None:
            self._theta = self.A_inv() @ self.b
        return self._theta

    def update(self, x: np.ndarray, reward: float) -> None:
        # A += x x^T;  b += r x
        # Use np.outer to keep things explicit.
        self.A += np.outer(x, x)
        self.b += reward * x
        self.pulls += 1
        self.rewards_sum += float(reward)
        # Invalidate caches.
        self._A_inv = None
        self._theta = None


class LinUCB:
    """LinUCB over a fixed set of arm ids.

    Arms are added lazily — first time we see arm_id we allocate its (A, b).
    Thread-safe for concurrent serve + update (one lock per arm; cheap).
    """

    def __init__(self, context_dim: int, alpha: float = 1.0) -> None:
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.context_dim = context_dim
        self.alpha = alpha
        self._arms: dict[int, ArmStats] = {}
        self._arm_locks: dict[int, threading.Lock] = {}
        self._registry_lock = threading.Lock()

    # --- internal ---

    def _arm(self, arm_id: int) -> tuple[ArmStats, threading.Lock]:
        # Fast path
        stats = self._arms.get(arm_id)
        if stats is not None:
            return stats, self._arm_locks[arm_id]
        # Allocate under registry lock
        with self._registry_lock:
            stats = self._arms.get(arm_id)
            if stats is None:
                stats = ArmStats.initialize(self.context_dim)
                self._arms[arm_id] = stats
                self._arm_locks[arm_id] = threading.Lock()
            return self._arms[arm_id], self._arm_locks[arm_id]

    # --- public ---

    def score(
        self,
        arm_id: int,
        context: np.ndarray,
        alpha: float | None = None,
    ) -> tuple[float, float, float]:
        """Return (total_score, exploit, explore) for one arm under context x."""
        if context.shape != (self.context_dim,):
            raise ValueError(
                f"context shape {context.shape} != ({self.context_dim},)"
            )
        a = alpha if alpha is not None else self.alpha
        stats, lock = self._arm(arm_id)
        with lock:
            theta = stats.theta()
            A_inv = stats.A_inv()
            exploit = float(theta @ context)
            # Confidence width: alpha * sqrt(x^T A^-1 x)
            width_sq = float(context @ A_inv @ context)
            # Guard against tiny negative values from FP error.
            explore = a * float(np.sqrt(max(width_sq, 0.0)))
        return exploit + explore, exploit, explore

    def update(self, arm_id: int, context: np.ndarray, reward: float) -> ArmStats:
        if context.shape != (self.context_dim,):
            raise ValueError(
                f"context shape {context.shape} != ({self.context_dim},)"
            )
        stats, lock = self._arm(arm_id)
        with lock:
            stats.update(context.astype(np.float64, copy=False), float(reward))
        return stats

    def arm_count(self) -> int:
        return len(self._arms)

    def total_pulls(self) -> int:
        return sum(s.pulls for s in self._arms.values())

    def snapshot(self) -> dict[int, dict[str, float]]:
        """Diagnostic snapshot: pulls + mean reward per arm."""
        out: dict[int, dict[str, float]] = {}
        for arm_id, stats in self._arms.items():
            with self._arm_locks[arm_id]:
                out[arm_id] = {
                    "pulls": stats.pulls,
                    "mean_reward": (stats.rewards_sum / stats.pulls) if stats.pulls else 0.0,
                }
        return out
