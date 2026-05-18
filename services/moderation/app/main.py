"""FastAPI entrypoint for the moderation service.

Endpoints:
    GET  /health
    POST /moderate            -> run text+visual+audio classifiers and fuse
    GET  /admin/snapshot      -> which models are loaded
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from shared.logging import configure_logging, log_event
from shared.schemas import HealthResponse

from .audio_classifier import AudioClassifier
from .config import Settings, get_settings
from .fusion import FusionWeights, fuse
from .schemas import HealthDetails, ModerationRequest, ModerationResponse
from .text_classifier import TextClassifier
from .visual_classifier import VisualClassifier


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(service=settings.service_name, level=settings.log_level)
    profile = settings.model_profile.value
    app.state.settings = settings
    app.state.text_clf = TextClassifier(profile=profile, model_name=settings.text_model_lite)
    app.state.visual_clf = VisualClassifier(profile=profile, model_name=settings.visual_model_prod)
    app.state.audio_clf = AudioClassifier(profile=profile)
    log_event(
        logger,
        "service_starting",
        service=settings.service_name,
        profile=profile,
    )
    yield
    log_event(logger, "service_stopping")


app = FastAPI(title="distrebute moderation", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s: Settings = app.state.settings
    return HealthResponse(service=s.service_name, model_profile=s.model_profile.value)


@app.post("/moderate", response_model=ModerationResponse)
def moderate(req: ModerationRequest) -> ModerationResponse:
    s: Settings = app.state.settings
    text_clf: TextClassifier = app.state.text_clf
    visual_clf: VisualClassifier = app.state.visual_clf
    audio_clf: AudioClassifier = app.state.audio_clf

    text_flags = text_clf.classify(req.transcript)
    visual_flags = visual_clf.classify(req.keyframes)
    audio_flags = audio_clf.classify(req.audio_segments)

    weights = FusionWeights(
        text=s.weight_text, visual=s.weight_visual, audio=s.weight_audio
    )
    result = fuse(
        text_flags + visual_flags + audio_flags,
        weights=weights,
        allow_below=s.allow_below,
        block_above=s.block_above,
    )

    log_event(
        logger,
        "moderation_decision",
        video_id=req.video_id,
        decision=result.decision,
        severity=result.fused_severity,
        n_flags=len(result.flags),
    )
    return ModerationResponse(
        video_id=req.video_id,
        decision=result.decision,
        severity=result.fused_severity,
        flags=result.flags,
        per_modality=result.per_modality,
    )


@app.get("/admin/snapshot", response_model=HealthDetails)
def admin_snapshot() -> HealthDetails:
    return HealthDetails(
        text_loaded=app.state.text_clf.is_loaded(),
        visual_loaded=app.state.visual_clf.is_loaded(),
        audio_loaded=app.state.audio_clf.is_loaded(),
    )
