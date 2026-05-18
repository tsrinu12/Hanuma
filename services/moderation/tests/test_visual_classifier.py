import numpy as np

from app.schemas import Keyframe
from app.visual_classifier import VisualClassifier, _hash_text_embed, _ANCHORS


def test_random_embedding_no_flag():
    """A neutral random embedding shouldn't match any anchor strongly."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal(256)
    v /= np.linalg.norm(v)
    clf = VisualClassifier(profile="lite", threshold=0.7)  # high threshold
    flags = clf.classify([Keyframe(t_s=1.0, embedding=v.tolist())])
    assert flags == []


def test_anchor_aligned_embedding_fires_matching_category():
    """If a frame's embedding equals an anchor's embedding, that anchor fires."""
    # Take the first anchor and craft a frame embedding identical to its
    # lite-mode text embedding.
    cat, prompt, _ = _ANCHORS[0]   # violence
    anchor_vec = _hash_text_embed(prompt, dim=128)
    clf = VisualClassifier(profile="lite", threshold=0.3)
    flags = clf.classify([Keyframe(t_s=12.5, embedding=anchor_vec.tolist())])
    assert any(f.category == cat for f in flags)
    fired = [f for f in flags if f.category == cat][0]
    # span recovered from t_s
    assert fired.start_s == 11.5
    assert fired.end_s == 13.5
    assert "12.5" in fired.explanation


def test_no_embedding_skipped_gracefully():
    clf = VisualClassifier(profile="lite")
    flags = clf.classify([Keyframe(t_s=2.0, embedding=None)])
    assert flags == []
