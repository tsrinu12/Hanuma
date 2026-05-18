"""Tests for shot synthesis and chapter assembly."""

import numpy as np

from app.chapter_pipeline import (
    assemble_chapters,
    synthesize_shots_from_transcript,
    transcript_for_span,
)
from app.schemas import Shot, TranscriptSegment


def _t(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start_s=start, end_s=end, text=text)


def test_synthesize_shots_splits_on_silence():
    transcript = [
        _t(0.0, 2.0, "hello"),
        _t(2.5, 4.0, "world"),
        _t(10.0, 11.0, "new topic"),   # 6s gap, triggers split
        _t(11.5, 13.0, "still talking"),
    ]
    shots = synthesize_shots_from_transcript(transcript, min_gap_s=2.5)
    assert len(shots) == 2
    assert shots[0].start_s == 0.0
    assert shots[0].end_s == 4.0
    assert shots[1].start_s == 10.0
    assert shots[1].end_s == 13.0


def test_transcript_for_span_picks_overlapping_segments():
    transcript = [
        _t(0.0, 5.0, "intro"),
        _t(5.0, 10.0, "middle"),
        _t(10.0, 15.0, "outro"),
    ]
    assert transcript_for_span(transcript, 4.0, 11.0) == "intro middle outro"
    assert transcript_for_span(transcript, 0.0, 4.9) == "intro"


def test_chapters_merge_similar_shots():
    shots = [Shot(shot_id=i, start_s=i * 10.0, end_s=(i + 1) * 10.0) for i in range(4)]
    transcript = [
        _t(0.0, 9.0, "we talk about pricing strategy"),
        _t(10.0, 19.0, "more pricing discussion"),
        _t(20.0, 29.0, "completely different sports topic"),
        _t(30.0, 39.0, "more sports content"),
    ]
    # Two clearly-different "visual" embeddings
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    shot_embs = np.array([a, a, b, b])
    chapters = assemble_chapters(
        shots, transcript, shot_embs,
        merge_distance=0.5,
        max_chapter_seconds=120.0,
    )
    assert len(chapters) == 2
    assert chapters[0].shot_ids == [0, 1]
    assert chapters[1].shot_ids == [2, 3]
    # title should be non-empty
    assert chapters[0].title.strip() != ""


def test_max_chapter_seconds_forces_split():
    shots = [Shot(shot_id=i, start_s=i * 60.0, end_s=(i + 1) * 60.0) for i in range(5)]
    transcript = [_t(i * 60.0, (i + 1) * 60.0 - 1.0, f"shot {i}") for i in range(5)]
    same = np.array([[1.0, 0.0]] * 5)
    chapters = assemble_chapters(
        shots, transcript, same,
        merge_distance=1.0,            # would merge everything
        max_chapter_seconds=120.0,     # but length cap kicks in
    )
    # 60s shots, 120s cap: groups of 2 -> 3 chapters [0,1] [2,3] [4]
    assert len(chapters) == 3
    assert chapters[0].shot_ids == [0, 1]
    assert chapters[1].shot_ids == [2, 3]
    assert chapters[2].shot_ids == [4]


def test_no_embeddings_no_visual_merge():
    """Without shot embeddings, only the length cap can split."""
    shots = [Shot(shot_id=i, start_s=i * 10.0, end_s=(i + 1) * 10.0) for i in range(3)]
    transcript = [_t(i * 10.0, (i + 1) * 10.0 - 0.5, f"text{i}") for i in range(3)]
    chapters = assemble_chapters(
        shots, transcript, None,
        merge_distance=0.25,
        max_chapter_seconds=600.0,
    )
    assert len(chapters) == 1
    assert chapters[0].shot_ids == [0, 1, 2]
