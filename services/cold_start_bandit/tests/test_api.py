"""HTTP layer smoke tests via the FastAPI TestClient."""

import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def _emb(seed: int, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    return v.tolist()


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "cold_start_bandit"


def test_index_and_rank_round_trip():
    with TestClient(app) as client:
        for i in range(6):
            r = client.post(
                "/index_video",
                json={"video_id": f"v{i}", "content_embedding": _emb(i)},
            )
            assert r.status_code == 200, r.text

        # rank with a user embedding similar to v0
        r = client.post(
            "/rank",
            json={
                "user_id": "u1",
                "user_embedding": _emb(0),
                "top_k": 3,
                "candidates": [
                    {
                        "video_id": f"v{i}",
                        "creator_id": f"creator_{i}",
                        "content_embedding": _emb(i),
                        "age_hours": 0.0,
                        "impressions_global": 0,
                    }
                    for i in range(6)
                ],
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["user_id"] == "u1"
        assert len(body["results"]) == 3
        # ranks should be 0,1,2
        assert [item["rank"] for item in body["results"]] == [0, 1, 2]


def test_reward_updates_arm():
    with TestClient(app) as client:
        client.post(
            "/index_video",
            json={"video_id": "vr", "content_embedding": _emb(42)},
        )
        # We need a context vector of size context_dim (default 64). For the
        # test we fabricate a plausible one.
        ctx = [0.0] * 64
        ctx[0] = 1.0
        r = client.post(
            "/reward",
            json={
                "user_id": "u1",
                "video_id": "vr",
                "context": ctx,
                "reward": 1.0,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["accepted"] is True
        assert body["arm_pulls"] >= 1


def test_reward_404s_for_unknown_video():
    with TestClient(app) as client:
        r = client.post(
            "/reward",
            json={
                "user_id": "u1",
                "video_id": "does-not-exist",
                "context": [0.0] * 64,
                "reward": 1.0,
            },
        )
        assert r.status_code == 404
