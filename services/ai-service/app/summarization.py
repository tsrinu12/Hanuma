"""Generate a short summary + chapter highlights from a transcript.

Default path: local Hugging Face BART. Optional path: OpenAI / Bedrock if OPENAI_API_KEY set.
"""
import logging
from functools import lru_cache
from typing import List

from .config import settings

log = logging.getLogger("ai.summarization")


@lru_cache(maxsize=1)
def _hf_pipeline():
    from transformers import pipeline
    return pipeline("summarization", model=settings.summary_model)


def _chunk(text: str, max_chars: int = 3500) -> List[str]:
    out, buf = [], ""
    for sent in text.replace("\n", " ").split(". "):
        if len(buf) + len(sent) > max_chars:
            out.append(buf.strip()); buf = sent + ". "
        else:
            buf += sent + ". "
    if buf.strip():
        out.append(buf.strip())
    return out or [text]


def summarize(text: str) -> dict:
    if not text.strip():
        return {"summary": "", "highlights": []}

    if settings.openai_api_key:
        try:
            import httpx
            r = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content":
                         "Summarise the transcript in 4-6 sentences, then list 3-5 timestamped highlights."},
                        {"role": "user", "content": text[:12000]},
                    ],
                    "temperature": 0.2,
                },
                timeout=60.0,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return {"summary": content, "highlights": []}
        except Exception as e:
            log.warning("OpenAI summary failed, falling back to local: %s", e)

    pipe = _hf_pipeline()
    chunks = _chunk(text)
    pieces = []
    for c in chunks[:4]:  # cap for latency
        try:
            res = pipe(c, max_length=140, min_length=40, do_sample=False)
            pieces.append(res[0]["summary_text"])
        except Exception as e:
            log.warning("chunk summary failed: %s", e)
    return {"summary": " ".join(pieces).strip(), "highlights": []}


def highlights_from_segments(segments: List[dict], top_k: int = 5) -> List[dict]:
    """Naive heuristic: pick segments with longest text — proxy for information density."""
    sorted_segs = sorted(segments, key=lambda s: len(s.get("text", "")), reverse=True)[:top_k]
    return [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in sorted(sorted_segs, key=lambda s: s["start"])
    ]
