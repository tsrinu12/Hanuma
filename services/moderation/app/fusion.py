"""Cross-modal fusion.

Inputs: per-modality flags, each with (modality, category, severity, span).

We produce:
  - per-category severity = weighted noisy-OR across modalities
    s_c = 1 - prod_m (1 - w_m * max_severity_in_category_for_modality)
  - top-line severity = max over categories
  - decision = ALLOW / REVIEW / BLOCK by thresholds

The noisy-OR fusion is the right shape here because we want cross-modal
*agreement* to push severity higher than any single modality could
(transcript says "kill" + visual shows a gun + audio shows a bang -> we
should decide BLOCK even if no single modality is fully confident).

Explanations are propagated unchanged from the firing classifiers so the
creator sees exactly which modality flagged what timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import Decision, ModalityFlag, Modality


@dataclass
class FusionWeights:
    text: float = 1.0
    visual: float = 1.0
    audio: float = 0.8


@dataclass
class FusionResult:
    decision: Decision
    fused_severity: float
    per_modality: dict[str, float]
    flags: list[ModalityFlag]


def fuse(
    flags: list[ModalityFlag],
    *,
    weights: FusionWeights,
    allow_below: float,
    block_above: float,
) -> FusionResult:
    """Combine per-modality flags into a final decision."""
    # Group by (category, modality).
    cat_modality_max: dict[str, dict[Modality, float]] = {}
    for f in flags:
        m = cat_modality_max.setdefault(f.category, {})
        if f.severity > m.get(f.modality, 0.0):
            m[f.modality] = f.severity

    # Per-category fused via noisy-OR over modalities.
    per_cat_fused: dict[str, float] = {}
    for cat, per_mod in cat_modality_max.items():
        prod = 1.0
        for mod, sev in per_mod.items():
            w = getattr(weights, mod)
            prod *= 1.0 - (w * sev)
        per_cat_fused[cat] = 1.0 - prod

    fused = max(per_cat_fused.values(), default=0.0)

    # Per-modality top line (max across all categories for that modality).
    per_modality_top: dict[str, float] = {"text": 0.0, "visual": 0.0, "audio": 0.0}
    for f in flags:
        if f.severity > per_modality_top[f.modality]:
            per_modality_top[f.modality] = f.severity

    if fused >= block_above:
        decision: Decision = "BLOCK"
    elif fused >= allow_below:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    # Sort flags by severity desc, keep top 25 for transport efficiency.
    ranked = sorted(flags, key=lambda f: -f.severity)[:25]

    return FusionResult(
        decision=decision,
        fused_severity=fused,
        per_modality=per_modality_top,
        flags=ranked,
    )
