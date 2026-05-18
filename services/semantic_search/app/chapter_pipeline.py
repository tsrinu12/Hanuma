"""Shot -> chapter pipeline.

Real prod pipeline:
  1. TransNetV2 over video frames -> shot boundaries
  2. Per-shot CLIP keyframe embeddings
  3. Agglomerative merge of adjacent shots when:
       - cosine distance(embeddings) < merge_threshold
       - merged chapter <= max_chapter_seconds
  4. Per-chapter chapter title: take the most-extractive sentence from the
     overlapping transcript, with the chapter's keyframe summary appended.

Lite fallback (no TransNetV2, no CLIP):
  - Synthesize shots from transcript pauses (gap > 2.5s -> new shot)
  - Skip the visual merge; merge by transcript-embedding cosine instead.

Either way, the function returns ``Chapter`` objects with stable ids that
the retrieval index can key off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .schemas import Chapter, Shot, TranscriptSegment


logger = logging.getLogger(__name__)


@dataclass
class AssembledChapter:
    chapter_id: int
    start_s: float
    end_s: float
    title: str
    summary: str
    shot_ids: list[int]
    text: str  # concatenated transcript covering this chapter


def synthesize_shots_from_transcript(
    transcript: list[TranscriptSegment],
    min_gap_s: float = 2.5,
) -> list[Shot]:
    """Lite-mode fallback when no shot boundaries are supplied.

    Splits the transcript wherever silence > min_gap_s. The result is not as
    precise as TransNetV2 but exercises the rest of the pipeline.
    """
    if not transcript:
        return []
    shots: list[Shot] = []
    start = transcript[0].start_s
    for i, seg in enumerate(transcript):
        nxt = transcript[i + 1] if i + 1 < len(transcript) else None
        if nxt is None or (nxt.start_s - seg.end_s) >= min_gap_s:
            shots.append(
                Shot(shot_id=len(shots), start_s=start, end_s=seg.end_s)
            )
            if nxt is not None:
                start = nxt.start_s
    return shots


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def transcript_for_span(
    transcript: list[TranscriptSegment], start_s: float, end_s: float
) -> str:
    """Return concatenated transcript text overlapping ``[start_s, end_s]``."""
    parts: list[str] = []
    for seg in transcript:
        if _overlap(seg.start_s, seg.end_s, start_s, end_s) > 0:
            parts.append(seg.text.strip())
    return " ".join(parts)


def assemble_chapters(
    shots: list[Shot],
    transcript: list[TranscriptSegment],
    shot_embeddings: np.ndarray | None,
    *,
    merge_distance: float,
    max_chapter_seconds: float,
) -> list[AssembledChapter]:
    """Agglomerative-merge adjacent shots into chapters.

    Merge criterion:
      - cosine distance between shot embeddings < ``merge_distance``
      - resulting chapter length <= ``max_chapter_seconds``
      - both shots have non-empty transcript (heuristic against merging
        unrelated commercial-break gaps with content)

    When ``shot_embeddings`` is None we still produce one chapter per shot but
    we'll merge purely by length cap.
    """
    if not shots:
        return []
    # Per-shot transcript prefetch.
    shot_text = [transcript_for_span(transcript, s.start_s, s.end_s) for s in shots]

    # Build groups by greedy left-to-right merge.
    groups: list[list[int]] = [[0]]
    for i in range(1, len(shots)):
        prev_last = groups[-1][-1]
        cur = i
        # Length cap on the resulting merged chapter.
        merged_start = shots[groups[-1][0]].start_s
        merged_end = shots[cur].end_s
        if merged_end - merged_start > max_chapter_seconds:
            groups.append([cur])
            continue
        # Visual similarity (if we have embeddings).
        if shot_embeddings is not None:
            a = shot_embeddings[prev_last]
            b = shot_embeddings[cur]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
            cos = float(a @ b / denom)
            dist = 1.0 - cos
            if dist > merge_distance:
                groups.append([cur])
                continue
        # Transcript: avoid merging silent-shot into content shot.
        if shot_text[cur].strip() == "" and shot_text[prev_last].strip() != "":
            groups.append([cur])
            continue
        groups[-1].append(cur)

    chapters: list[AssembledChapter] = []
    for cid, group in enumerate(groups):
        start = shots[group[0]].start_s
        end = shots[group[-1]].end_s
        text = transcript_for_span(transcript, start, end)
        title = _make_title(text, fallback=f"Chapter {cid + 1}")
        summary = _first_two_sentences(text)
        chapters.append(
            AssembledChapter(
                chapter_id=cid,
                start_s=start,
                end_s=end,
                title=title,
                summary=summary,
                shot_ids=[shots[i].shot_id for i in group],
                text=text,
            )
        )
    return chapters


def _make_title(text: str, fallback: str, max_words: int = 8) -> str:
    """Cheap extractive title: first noun-y looking sentence, truncated.

    Real prod path: pass `text` to a small generator (e.g. T5-small) for a
    learned title. Lite mode keeps this dependency-free.
    """
    text = text.strip()
    if not text:
        return fallback
    # First sentence.
    for sep in (". ", "? ", "! "):
        if sep in text:
            first = text.split(sep, 1)[0]
            break
    else:
        first = text
    words = first.split()
    if len(words) > max_words:
        first = " ".join(words[:max_words]) + "..."
    return first


def _first_two_sentences(text: str) -> str:
    sents: list[str] = []
    buf = text
    for _ in range(2):
        for sep in (". ", "? ", "! "):
            if sep in buf:
                s, buf = buf.split(sep, 1)
                sents.append(s.strip())
                break
        else:
            if buf.strip():
                sents.append(buf.strip())
                buf = ""
            break
    return ". ".join(sents)


def to_api_chapters(assembled: list[AssembledChapter]) -> list[Chapter]:
    return [
        Chapter(
            chapter_id=c.chapter_id,
            start_s=c.start_s,
            end_s=c.end_s,
            title=c.title,
            summary=c.summary,
            shot_ids=c.shot_ids,
        )
        for c in assembled
    ]
