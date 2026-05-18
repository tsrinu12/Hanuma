"""VAD chunker tests."""

import numpy as np

from app.vad import VadChunker


SR = 16000


def _tone(duration_s: float, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(duration_s: float) -> np.ndarray:
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def _new_chunker():
    return VadChunker(
        sample_rate=SR,
        chunk_target_s=2.5,
        silence_break_s=0.4,
        min_voiced_s=0.5,
        energy_threshold=0.01,
    )


def test_silence_emits_nothing():
    ch = _new_chunker()
    chunks = ch.push(_silence(2.0))
    assert chunks == []
    assert ch.flush() is None


def test_speech_then_silence_closes_chunk():
    ch = _new_chunker()
    # 1s tone, then 1s silence -> chunk closes via silence break.
    chunks = ch.push(_tone(1.0))
    # not enough silence yet
    chunks2 = ch.push(_silence(1.0))
    total = chunks + chunks2
    assert len(total) == 1
    chunk = total[0]
    assert chunk.chunk_id == 0
    assert chunk.end_s > chunk.start_s
    assert chunk.samples.size > 0


def test_target_length_closes_chunk_even_without_silence():
    ch = _new_chunker()
    # Continuous 4s tone, no silence -> should close around 2.5s target.
    chunks = ch.push(_tone(4.0))
    assert len(chunks) >= 1
    # First chunk should be around 2.5s long (allow some slack).
    first = chunks[0]
    duration = first.end_s - first.start_s
    assert 2.0 <= duration <= 3.5


def test_chunk_ids_increment():
    ch = _new_chunker()
    out = []
    out += ch.push(_tone(1.0))
    out += ch.push(_silence(0.5))
    out += ch.push(_tone(1.0))
    out += ch.push(_silence(0.5))
    flushed = ch.flush()
    if flushed is not None:
        out.append(flushed)
    ids = [c.chunk_id for c in out]
    assert ids == list(range(len(out)))


def test_too_short_speech_doesnt_emit():
    """200ms of speech is below min_voiced_s and should not produce a chunk."""
    ch = _new_chunker()
    out = ch.push(_tone(0.2))
    out += ch.push(_silence(1.0))
    assert out == []
