"""Speech-to-text via faster-whisper (a fast CTranslate2 port of OpenAI Whisper)."""
import logging
from functools import lru_cache
from typing import List, Tuple

from faster_whisper import WhisperModel

from .config import settings

log = logging.getLogger("ai.transcription")


@lru_cache(maxsize=1)
def _model() -> WhisperModel:
    log.info("Loading Whisper model %s", settings.whisper_model)
    # On CPU you can switch compute_type="int8". On GPU use "float16".
    return WhisperModel(settings.whisper_model, device="auto", compute_type="int8")


def transcribe(audio_or_video_path: str, language: str | None = None) -> dict:
    model = _model()
    segments, info = model.transcribe(
        audio_or_video_path,
        language=language,
        vad_filter=True,
        beam_size=1,
    )
    seg_list: List[dict] = []
    full = []
    for s in segments:
        seg_list.append({"start": s.start, "end": s.end, "text": s.text.strip()})
        full.append(s.text.strip())
    return {
        "language": info.language,
        "full_text": " ".join(full),
        "segments": seg_list,
    }
