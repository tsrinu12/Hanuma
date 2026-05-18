"""HTTP/WebSocket smoke tests."""

import struct

import numpy as np
from fastapi.testclient import TestClient

from app.main import app


SR = 16000


def _tone(duration_s: float, amp: float = 0.3) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["service"] == "live_moderation"


def test_buffer_endpoint_emits_transcripts():
    samples = np.concatenate([_tone(1.0), np.zeros(SR, dtype=np.float32)]).tolist()
    with TestClient(app) as client:
        r = client.post("/moderate_buffer", json={
            "stream_id": "t1",
            "sample_rate": SR,
            "samples": samples,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert any(e["event"] == "TRANSCRIPT" for e in body["events"])


def test_websocket_round_trip():
    """Open WS, send a control frame + an audio frame, expect a STATUS then events."""
    with TestClient(app) as client:
        with client.websocket_connect("/ws/moderate") as ws:
            # First message from server: STATUS ready
            first = ws.receive_text()
            assert "STATUS" in first

            # Send control frame.
            ws.send_text('{"stream_id":"t1","sample_rate":16000}')
            # Send a binary audio frame (1s of tone, then 1s silence).
            buf = np.concatenate([_tone(1.0), np.zeros(SR, dtype=np.float32)])
            ws.send_bytes(struct.pack(f"<{buf.size}f", *buf))

            # Expect at least one TRANSCRIPT event.
            got_transcript = False
            for _ in range(5):
                msg = ws.receive_text()
                if "TRANSCRIPT" in msg:
                    got_transcript = True
                    break
            assert got_transcript
