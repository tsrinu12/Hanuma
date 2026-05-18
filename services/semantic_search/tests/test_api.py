"""End-to-end HTTP test through the FastAPI TestClient."""

from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["service"] == "semantic_search"


def test_index_then_search():
    payload = {
        "video_id": "demo1",
        "duration_s": 60.0,
        "transcript": [
            {"start_s": 0.0, "end_s": 5.0, "text": "Today we discuss pricing strategy for SaaS startups"},
            {"start_s": 5.5, "end_s": 12.0, "text": "Tiered pricing and freemium are the two main models"},
            {"start_s": 18.0, "end_s": 25.0, "text": "Now switching topics to onboarding flows for new users"},
            {"start_s": 25.5, "end_s": 35.0, "text": "Onboarding determines whether a new signup converts"},
            {"start_s": 50.0, "end_s": 58.0, "text": "Finally lets talk about churn reduction tactics"},
        ],
        # No shots provided -> service synthesizes from transcript gaps
    }
    with TestClient(app) as client:
        r = client.post("/index_video", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["video_id"] == "demo1"
        assert body["chunks_indexed"] >= 1
        assert len(body["chapters"]) >= 1

        # Query for the part about onboarding.
        r2 = client.post("/search", json={"query": "onboarding flows", "top_k": 1})
        assert r2.status_code == 200
        hits = r2.json()["hits"]
        assert len(hits) == 1
        assert hits[0]["video_id"] == "demo1"
        # The onboarding chapter starts at 18s; the matching chapter should
        # cover that timestamp.
        assert hits[0]["start_s"] <= 25.0
        assert hits[0]["end_s"] >= 18.0


def test_search_video_id_filter():
    with TestClient(app) as client:
        for vid in ("a", "b"):
            client.post("/index_video", json={
                "video_id": vid,
                "duration_s": 30.0,
                "transcript": [
                    {"start_s": 0.0, "end_s": 5.0, "text": "pricing strategy"},
                ],
            })
        r = client.post("/search", json={"query": "pricing", "video_id": "a", "top_k": 3})
        assert r.status_code == 200
        for hit in r.json()["hits"]:
            assert hit["video_id"] == "a"
