"""End-to-end HTTP tests through TestClient."""

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.visual_classifier import _hash_text_embed, _ANCHORS


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["service"] == "moderation"


def test_clean_video_is_allow():
    payload = {
        "video_id": "clean1",
        "duration_s": 30.0,
        "transcript": [{"start_s": 0.0, "end_s": 30.0, "text": "a nice talk about gardening"}],
    }
    with TestClient(app) as client:
        r = client.post("/moderate", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["decision"] == "ALLOW"
        assert body["flags"] == []
        assert body["severity"] == 0.0


def test_textual_harassment_is_block():
    payload = {
        "video_id": "harass1",
        "duration_s": 5.0,
        "transcript": [{"start_s": 0.0, "end_s": 5.0, "text": "kys"}],
    }
    with TestClient(app) as client:
        r = client.post("/moderate", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["decision"] == "BLOCK"
        assert any(
            f["modality"] == "text" and f["category"] == "harassment"
            for f in body["flags"]
        )


def test_cross_modal_boost_pushes_to_block():
    """Mild text + mild visual at the same category fuse past block threshold."""
    cat, prompt, _ = _ANCHORS[0]   # violence
    anchor = _hash_text_embed(prompt, dim=128)
    payload = {
        "video_id": "cm1",
        "duration_s": 10.0,
        "transcript": [{"start_s": 0.0, "end_s": 10.0, "text": "I want to kill them"}],
        "keyframes": [{"t_s": 5.0, "embedding": anchor.tolist()}],
    }
    with TestClient(app) as client:
        r = client.post("/moderate", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["decision"] in ("REVIEW", "BLOCK")
        # Both modalities should have non-zero severity contributions.
        assert body["per_modality"]["text"] > 0.0
        assert body["per_modality"]["visual"] > 0.0


def test_explanations_are_present():
    payload = {
        "video_id": "exp1",
        "duration_s": 5.0,
        "transcript": [{"start_s": 0.0, "end_s": 5.0, "text": "you are stupid"}],
    }
    with TestClient(app) as client:
        body = client.post("/moderate", json=payload).json()
        assert all("transcript:" in f["explanation"] for f in body["flags"])


def test_admin_snapshot():
    with TestClient(app) as client:
        body = client.get("/admin/snapshot").json()
        # Lite-mode: text + visual + audio all consider themselves "loaded".
        assert body["text_loaded"] is True
        assert body["visual_loaded"] is True
        assert body["audio_loaded"] is True
