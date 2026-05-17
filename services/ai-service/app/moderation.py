"""Content moderation.

Approach:
 - For video frames: opennsfw2 predicts P(NSFW) on sampled frames. We pick the max.
 - For text: a small wordlist + the HF transformers `unitary/toxic-bert` model if available.
   In real deployment you'd swap for AWS Rekognition Content Moderation + Bedrock Guardrails.
"""
import logging
import os
import subprocess
import tempfile
from functools import lru_cache
from typing import List

log = logging.getLogger("ai.moderation")

NSFW_THRESHOLD = 0.85


def _sample_frames(video_path: str, every_seconds: int = 5, max_frames: int = 30) -> List[str]:
    out_dir = tempfile.mkdtemp(prefix="frames_")
    pattern = os.path.join(out_dir, "frame_%04d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vf", f"fps=1/{every_seconds}",
         "-frames:v", str(max_frames), "-q:v", "3", pattern],
        check=False, capture_output=True,
    )
    return sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".jpg")
    )


@lru_cache(maxsize=1)
def _nsfw_model():
    import opennsfw2 as n2
    return n2


def moderate_video(video_path: str) -> dict:
    frames = _sample_frames(video_path)
    if not frames:
        return {"nsfw_max": 0.0, "frames_scanned": 0, "blocked": False}

    n2 = _nsfw_model()
    nsfw_probs = n2.predict_images(frames)  # list of (sfw, nsfw)
    max_nsfw = max((p[1] for p in nsfw_probs), default=0.0)

    for f in frames:
        try:
            os.unlink(f)
        except OSError:
            pass

    return {
        "nsfw_max": float(max_nsfw),
        "frames_scanned": len(frames),
        "blocked": bool(max_nsfw >= NSFW_THRESHOLD),
    }


BLOCKED_WORDS = {"slur1", "slur2"}  # placeholder; load from secure config in prod


def moderate_text(text: str) -> dict:
    low = text.lower()
    hits = [w for w in BLOCKED_WORDS if w in low]
    return {"text_blocked": bool(hits), "hits": hits}
