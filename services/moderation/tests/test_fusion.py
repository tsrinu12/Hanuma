from app.fusion import FusionWeights, fuse
from app.schemas import ModalityFlag


def _f(modality, category, severity, start=0.0, end=1.0, explanation="x"):
    return ModalityFlag(
        modality=modality, category=category, severity=severity,
        start_s=start, end_s=end, explanation=explanation,
    )


def test_no_flags_is_allow():
    w = FusionWeights()
    res = fuse([], weights=w, allow_below=0.35, block_above=0.85)
    assert res.decision == "ALLOW"
    assert res.fused_severity == 0.0


def test_single_low_severity_is_allow():
    w = FusionWeights()
    res = fuse(
        [_f("text", "toxicity", 0.2)],
        weights=w, allow_below=0.35, block_above=0.85,
    )
    assert res.decision == "ALLOW"


def test_single_high_severity_is_block():
    w = FusionWeights()
    res = fuse(
        [_f("text", "harassment", 0.95)],
        weights=w, allow_below=0.35, block_above=0.85,
    )
    assert res.decision == "BLOCK"


def test_cross_modal_agreement_boosts_severity():
    """Two modalities each at 0.6 should fuse to higher than 0.6."""
    w = FusionWeights(text=1.0, visual=1.0, audio=1.0)
    res = fuse(
        [
            _f("text", "violence", 0.6),
            _f("visual", "violence", 0.6),
        ],
        weights=w, allow_below=0.35, block_above=0.85,
    )
    # noisy-OR: 1 - (1 - 0.6)*(1 - 0.6) = 1 - 0.16 = 0.84
    assert 0.83 < res.fused_severity < 0.85


def test_categories_dont_combine_across_categories():
    """Distinct categories don't add up - each is fused independently."""
    w = FusionWeights()
    res = fuse(
        [
            _f("text", "toxicity", 0.5),
            _f("visual", "sexual", 0.5),
        ],
        weights=w, allow_below=0.35, block_above=0.85,
    )
    # Max per-category is just 0.5 (no cross-modal agreement within a category).
    assert res.fused_severity == 0.5


def test_per_modality_aggregated():
    w = FusionWeights()
    res = fuse(
        [
            _f("text", "toxicity", 0.3),
            _f("text", "toxicity", 0.7),
            _f("visual", "violence", 0.4),
        ],
        weights=w, allow_below=0.35, block_above=0.85,
    )
    assert res.per_modality["text"] == 0.7
    assert res.per_modality["visual"] == 0.4
    assert res.per_modality["audio"] == 0.0


def test_flags_sorted_by_severity_desc():
    w = FusionWeights()
    res = fuse(
        [
            _f("text", "toxicity", 0.3),
            _f("text", "toxicity", 0.9),
            _f("visual", "violence", 0.6),
        ],
        weights=w, allow_below=0.35, block_above=0.85,
    )
    sevs = [f.severity for f in res.flags]
    assert sevs == sorted(sevs, reverse=True)
