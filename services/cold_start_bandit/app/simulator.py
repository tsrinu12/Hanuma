"""Bandit simulator — synthetic environment for regression-testing changes.

The simulator builds:
  - N synthetic peer clusters, each with a hidden theta_true
  - M synthetic videos, each assigned to a cluster, with a content embedding
    near the cluster's "center" embedding
  - synthetic users with embeddings drawn from a mixture over those centers

A reward is drawn as sigmoid(theta_true_cluster . context) + small noise. The
bandit converges to high cumulative reward when LinUCB is correct and exposes
regressions when it isn't.

Used by the test suite; can also be invoked directly:

    python -m app.simulator --rounds 5000 --alpha 1.0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .bandit import LinUCB
from .features import build_context_vector
from .peer_cluster import PeerClusterIndex


@dataclass
class SimConfig:
    n_clusters: int = 8
    videos_per_cluster: int = 20
    embed_dim: int = 32
    context_dim: int = 64
    alpha: float = 1.0
    rounds: int = 2000
    seed: int = 0


def run_simulation(cfg: SimConfig) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed)

    # 1) synthetic cluster centers in embedding space
    centers = rng.normal(size=(cfg.n_clusters, cfg.embed_dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12

    # 2) hidden ground-truth theta per cluster, in *context* space
    theta_true = rng.normal(size=(cfg.n_clusters, cfg.context_dim)) * 0.5

    # 3) videos: each near a cluster center
    videos: list[tuple[str, int, np.ndarray]] = []  # (video_id, cluster, embed)
    for c in range(cfg.n_clusters):
        for v in range(cfg.videos_per_cluster):
            e = centers[c] + 0.1 * rng.normal(size=cfg.embed_dim)
            e /= np.linalg.norm(e) + 1e-12
            videos.append((f"v_c{c}_n{v}", c, e))

    # 4) index videos in the peer cluster store (and let it cluster them)
    pc = PeerClusterIndex(dim=cfg.embed_dim, n_clusters=cfg.n_clusters)
    for vid, _, e in videos:
        pc.add(vid, e)
    pc.recluster(n_clusters=cfg.n_clusters)

    # 5) bandit
    bandit = LinUCB(context_dim=cfg.context_dim, alpha=cfg.alpha)

    # 6) random user generator: pick a center, perturb
    def sample_user() -> tuple[np.ndarray, int]:
        c = int(rng.integers(0, cfg.n_clusters))
        e = centers[c] + 0.2 * rng.normal(size=cfg.embed_dim)
        e /= np.linalg.norm(e) + 1e-12
        return e, c

    # 7) simulate
    cum_reward = 0.0
    optimal_reward = 0.0
    for t in range(cfg.rounds):
        u, _ = sample_user()
        # Sample a batch of candidate videos
        idxs = rng.choice(len(videos), size=20, replace=False)
        cands = [videos[i] for i in idxs]

        # Score each via bandit
        best_score = -np.inf
        best_choice = None
        best_context = None
        best_cluster = None
        # also track the oracle optimum for regret
        oracle_best_reward = -np.inf
        for vid, true_c, e in cands:
            ctx = build_context_vector(
                user_embedding=u,
                content_embedding=e,
                age_hours=0.0,
                impressions_global=0,
                context_dim=cfg.context_dim,
            )
            arm_id = pc.cluster_for(vid) or 0
            score, _, _ = bandit.score(arm_id, ctx)
            if score > best_score:
                best_score = score
                best_choice = vid
                best_context = ctx
                best_cluster = arm_id

            # oracle: best truly-expected reward
            mean = float(theta_true[true_c] @ ctx)
            r = _sigmoid(mean)
            if r > oracle_best_reward:
                oracle_best_reward = r

        # observe reward for the bandit's pick
        # (use the true cluster of the chosen video for ground truth)
        chosen_true_c = next(true_c for vid, true_c, _ in cands if vid == best_choice)
        mean_reward = float(theta_true[chosen_true_c] @ best_context)
        observed = _sigmoid(mean_reward) + 0.05 * rng.standard_normal()
        observed = float(np.clip(observed, 0.0, 1.0))

        bandit.update(best_cluster, best_context, observed)
        cum_reward += observed
        optimal_reward += oracle_best_reward

    return {
        "rounds": float(cfg.rounds),
        "cumulative_reward": float(cum_reward),
        "oracle_reward": float(optimal_reward),
        "regret_ratio": float((optimal_reward - cum_reward) / max(optimal_reward, 1e-9)),
        "arms_used": float(bandit.arm_count()),
    }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    res = run_simulation(SimConfig(rounds=args.rounds, alpha=args.alpha, seed=args.seed))
    print(res)
