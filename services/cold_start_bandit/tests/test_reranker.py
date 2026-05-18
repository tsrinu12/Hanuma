"""MMR reranker tests."""

import numpy as np
import pytest

from app.reranker import CandidateForRerank, mmr_rerank


def _cand(vid: str, creator: str, rel: float, emb: np.ndarray) -> CandidateForRerank:
    return CandidateForRerank(video_id=vid, creator_id=creator, relevance=rel, embedding=emb)


def test_pure_relevance_orders_by_score():
    """lambda=1.0 reduces to top-k by relevance."""
    e = np.array([1.0, 0.0])
    cands = [
        _cand("a", "c1", 0.1, e),
        _cand("b", "c2", 0.9, e),
        _cand("c", "c3", 0.5, e),
    ]
    order = mmr_rerank(cands, top_k=3, lambda_=1.0, creator_penalty=0.0)
    picked_ids = [cands[i].video_id for i in order]
    assert picked_ids == ["b", "c", "a"]


def test_diversity_breaks_near_duplicates():
    """With lambda < 1, the second pick should not be a near-duplicate of the first."""
    e_main = np.array([1.0, 0.0, 0.0])
    e_dup = np.array([0.99, 0.01, 0.0])     # near duplicate of e_main
    e_diff = np.array([0.0, 1.0, 0.0])      # orthogonal
    cands = [
        _cand("main", "c1", 1.0, e_main),
        _cand("dup", "c2", 0.95, e_dup),
        _cand("diff", "c3", 0.6, e_diff),
    ]
    order = mmr_rerank(cands, top_k=2, lambda_=0.5, creator_penalty=0.0)
    picked = [cands[i].video_id for i in order]
    assert picked[0] == "main"
    # With heavy diversity weight, the orthogonal one wins despite lower relevance.
    assert picked[1] == "diff"


def test_creator_penalty_suppresses_repeats():
    e = np.array([1.0, 0.0])
    cands = [
        _cand("a", "creator1", 1.0, e),
        _cand("b", "creator1", 0.9, e),     # same creator
        _cand("c", "creator2", 0.85, e),    # different creator
    ]
    order = mmr_rerank(cands, top_k=2, lambda_=1.0, creator_penalty=0.5)
    picked = [cands[i].video_id for i in order]
    assert picked[0] == "a"
    # creator1 already represented; large penalty should push creator2 ahead.
    assert picked[1] == "c"


def test_top_k_caps_output():
    e = np.array([1.0, 0.0])
    cands = [_cand(str(i), str(i), 1.0 - 0.01 * i, e) for i in range(5)]
    order = mmr_rerank(cands, top_k=3)
    assert len(order) == 3


def test_empty_input():
    assert mmr_rerank([], top_k=5) == []


def test_invalid_lambda():
    with pytest.raises(ValueError):
        mmr_rerank([_cand("a", "x", 1.0, np.array([1.0]))], top_k=1, lambda_=1.5)
