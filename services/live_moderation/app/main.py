"""FastAPI app for the live moderation service.

Endpoints:
    GET  /health                       liveness
    WS   /ws/moderate                  bidirectional moderation stream
    POST /moderate_buffer              one-shot synchronous helper (testing)
"""

from __future__ import annotations

import json
import logging
import struct
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from shared.logging import configure_logging, log_event
from shared.schemas import HealthResponse

from .asr import WhisperASR
from .config import Settings, get_settings
from .pipeline import LivePipeline
from .schemas import StatusEvent
from .toxicity import StreamingToxicity


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(service=settings.service_name, level=settings.log_level)
    profile = settings.model_profile.value
    app.state.settings = settings
    # ASR and toxicity classifiers are shared across streams (thread-safe).
    asr_model = settings.asr_model_lite if profile == "lite" else settings.asr_model_prod
    app.state.asr = WhisperASR(
        model_name=asr_model, sample_rate=settings.asr_sample_rate, profile=profile
    )
    app.state.toxicity = StreamingToxicity(profile=profile)
    log_event(
        logger,
        "service_starting",
        service=settings.service_name,
        profile=profile,
        asr_model=asr_model,
    )
    yield
    log_event(logger, "service_stopping")


app = FastAPI(title="distrebute live moderation", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s: Settings = app.state.settings
    return HealthResponse(service=s.service_name, model_profile=s.model_profile.value)


# ------------------ WebSocket ------------------


@app.websocket("/ws/moderate")
async def ws_moderate(ws: WebSocket) -> None:
    """Stream protocol:

    1. Client opens the socket. Server sends a STATUS event:
         {"event": "STATUS", "msg": "ready"}
    2. Client sends a JSON text frame with stream parameters (optional):
         {"stream_id": "abc", "sample_rate": 16000}
       If omitted, defaults to 16000 mono.
    3. Client streams audio as binary frames. Each binary frame is
       float32 little-endian PCM mono (one float per sample).
    4. Server emits TRANSCRIPT and FLAG events as JSON text frames.
    5. Either side can close. On close, server flushes the pipeline and
       sends any pending events.
    """
    settings: Settings = app.state.settings
    await ws.accept()
    await ws.send_text(StatusEvent(msg="ready").model_dump_json())

    sample_rate = settings.asr_sample_rate
    stream_id = "anonymous"

    # Optionally consume a control frame.
    try:
        first = await ws.receive()
    except WebSocketDisconnect:
        return
    if first.get("type") == "websocket.disconnect":
        return
    if first.get("text") is not None:
        try:
            ctrl = json.loads(first["text"])
            stream_id = str(ctrl.get("stream_id", stream_id))
            sample_rate = int(ctrl.get("sample_rate", sample_rate))
        except (ValueError, TypeError):
            pass
        try:
            first = await ws.receive()
        except WebSocketDisconnect:
            return

    pipeline = LivePipeline.for_settings(
        settings=settings,
        sample_rate=sample_rate,
        asr=app.state.asr,
        toxicity=app.state.toxicity,
    )
    log_event(logger, "stream_open", stream_id=stream_id, sample_rate=sample_rate)

    try:
        # Process the binary frame we already pulled (`first`), then loop.
        msg = first
        while True:
            if msg.get("type") == "websocket.disconnect":
                break
            bytes_payload = msg.get("bytes")
            if bytes_payload:
                samples = _decode_float32(bytes_payload)
                for event in pipeline.push(samples):
                    await ws.send_text(event.model_dump_json())
            msg = await ws.receive()
    except WebSocketDisconnect:
        pass
    finally:
        # Drain.
        for event in pipeline.flush():
            try:
                await ws.send_text(event.model_dump_json())
            except Exception:
                break
        log_event(logger, "stream_close", stream_id=stream_id)


def _decode_float32(b: bytes) -> np.ndarray:
    """Decode little-endian float32 PCM bytes -> numpy array."""
    if not b:
        return np.zeros(0, dtype=np.float32)
    n = len(b) // 4
    return np.array(struct.unpack(f"<{n}f", b[: n * 4]), dtype=np.float32)


# ------------------ Buffered (testing) ------------------


class BufferRequest(BaseModel):
    stream_id: str = "buffer"
    sample_rate: int = 16000
    samples: list[float]


class BufferResponse(BaseModel):
    events: list[dict]


@app.post("/moderate_buffer", response_model=BufferResponse)
def moderate_buffer(req: BufferRequest) -> BufferResponse:
    """Run the full pipeline over a single in-memory buffer.

    Useful for tests and for batch-moderating recorded audio with the same
    code path as the live stream.
    """
    settings: Settings = app.state.settings
    pipeline = LivePipeline.for_settings(
        settings=settings,
        sample_rate=req.sample_rate,
        asr=app.state.asr,
        toxicity=app.state.toxicity,
    )
    samples = np.asarray(req.samples, dtype=np.float32)
    events = []
    for ev in pipeline.push(samples):
        events.append(ev.model_dump())
    for ev in pipeline.flush():
        events.append(ev.model_dump())
    return BufferResponse(events=events)
