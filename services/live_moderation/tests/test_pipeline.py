"""End-to-end pipeline test using the in-memory buffer endpoint shape."""

import numpy as np

from app.asr import WhisperASR
from app.config import get_settings
from app.pipeline import LivePipeline
from app.toxicity import StreamingToxicity


SR = 16000


def _tone(duration_s: float, amp: float = 0.3) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def _build_pipeline(monkeypatch_lite_text=None):
    settings = get_settings()
    asr = WhisperASR(model_name="tiny.en", sample_rate=SR, profile="lite")
    tox = StreamingToxicity(profile="lite")
    pipeline = LivePipeline.for_settings(
        settings=settings, sample_rate=SR, asr=asr, toxicity=tox
    )
    if monkeypatch_lite_text is not None:
        # Patch the ASR's lite transcriber to emit a fixed string so
        # toxicity has something concrete to operate on.
        def fake(samples):
            from app.asr import TranscriptResult
            return TranscriptResult(text=monkeypatch_lite_text, language="en")
        asr._transcribe_lite = fake  # type: ignore[method-assign]
    return pipeline


def test_pipeline_emits_transcript_per_chunk():
    pipeline = _build_pipeline()
    events = []
    for ev in pipeline.push(_tone(1.0)):
        events.append(ev)
    for ev in pipeline.push(_silence(1.0)):
        events.append(ev)
    for ev in pipeline.flush():
        events.append(ev)
    # We should have at least one TRANSCRIPT event.
    transcripts = [e for e in events if e.event == "TRANSCRIPT"]
    assert len(transcripts) >= 1


def test_pipeline_emits_flag_when_toxicity_fires():
    pipeline = _build_pipeline(monkeypatch_lite_text="you stupid moron kys")
    events = []
    for ev in pipeline.push(_tone(1.0)):
        events.append(ev)
    for ev in pipeline.push(_silence(1.0)):
        events.append(ev)
    for ev in pipeline.flush():
        events.append(ev)
    flags = [e for e in events if e.event == "FLAG"]
    assert len(flags) >= 1
    assert flags[0].action in ("WARN", "BLOCK")
    assert flags[0].severity > 0.0


def test_pipeline_no_flag_for_clean_audio():
    pipeline = _build_pipeline(monkeypatch_lite_text="this is a perfectly normal phrase")
    events = []
    for ev in pipeline.push(_tone(1.0)):
        events.append(ev)
    for ev in pipeline.push(_silence(1.0)):
        events.append(ev)
    flags = [e for e in events if e.event == "FLAG"]
    assert flags == []
