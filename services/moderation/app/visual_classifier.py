"""Visual moderation.

Operates on keyframes (one frame every ~2s by default).

Lite mode: classifies from a CLIP-style embedding using a built-in set of
text-prompt anchors. We compute cosine similarity between the keyframe
embedding and the anchor's text embedding (approximated in lite mode via a
deterministic hash embedder). The category with the highest sim above a
threshold becomes the flag.

Prod mode: real CLIP. Same anchor concept, but text embeddings come from
CLIP's text tower so the cosines are meaningful.

This module is intentionally embedding-based: the upload pipeline computes
keyframe embeddings once (with the same encoder used for recommendation)
and we reuse them. That way moderation does not need its own GPU pool.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass

import numpy as np

from .schemas import Category, Keyframe, ModalityFlag


logger = logging.getLogger(__name__)


# Anchor prompts, mapped to our category vocabulary. The visual model checks
# similarity against these prompts; any one passing threshold flags the frame.
_ANCHORS: list[tuple[Category, str, str]] = [
    ("violence", "a violent fight with weapons or blood", "violent imagery"),
    ("violence", "a person being physically attacked", "depicted assault"),
    ("sexual",   "explicit nude sexual content",          "explicit sexual content"),
    ("sexual",   "partial nudity in a suggestive pose",   "suggestive nudity"),
    ("self_harm","a person harming themselves",           "self-harm imagery"),
    ("hate",     "a hate-group symbol or flag",           "hate symbol detected"),
]


def _hash_text_embed(text: str, dim: int) -> np.ndarray:
    """Deterministic, dependency-free text embedder for lite mode.

    Maps tokens to signed-hash buckets. Good enough for unit tests; useless
    for real moderation.
    """
    out = np.zeros(dim, dtype=np.float64)
    for tok in text.lower().split():
        h = hashlib.blake2b(tok.encode(), digest_size=16).digest()
        for i in range(min(dim, 16)):
            bucket = i % dim
            sign = 1.0 if (h[i] & 1) else -1.0
            out[bucket] += sign * ((h[i] >> 1) & 0x3F) / 64.0
    n = float(np.linalg.norm(out))
    return out if n == 0.0 else out / n


@dataclass
class VisualClassifier:
    profile: str = "lite"
    model_name: str = "openai/clip-vit-base-patch32"
    # threshold on cosine similarity for a flag to fire
    threshold: float = 0.45
    _clip = None
    _processor = None
    _anchor_embeds: dict[int, np.ndarray] = None  # type: ignore[assignment]
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._anchor_embeds = {}
        self._lock = threading.Lock()

    def is_loaded(self) -> bool:
        return self.profile == "lite" or self._clip is not None

    # --- prod ---

    def _load_prod(self) -> None:
        with self._lock:
            if self._clip is not None:
                return
            try:
                from transformers import CLIPModel, CLIPProcessor  # type: ignore
            except ImportError as e:
                raise RuntimeError("transformers not installed for prod") from e
            logger.info("loading CLIP: %s", self.model_name)
            self._clip = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._clip.eval()

    # --- anchor embeds (lazy) ---

    def _anchor_embed(self, idx: int, prompt: str, dim: int) -> np.ndarray:
        v = self._anchor_embeds.get(idx)
        if v is not None:
            return v
        if self.profile == "lite":
            v = _hash_text_embed(prompt, dim)
        else:
            import torch  # type: ignore

            self._load_prod()
            with torch.no_grad():
                inputs = self._processor(text=[prompt], return_tensors="pt", padding=True)  # type: ignore[union-attr]
                vec = self._clip.get_text_features(**inputs)[0].cpu().numpy()  # type: ignore[union-attr]
                vec /= np.linalg.norm(vec) + 1e-12
                v = vec.astype(np.float64)
        self._anchor_embeds[idx] = v
        return v

    # --- public ---

    def classify(self, keyframes: list[Keyframe]) -> list[ModalityFlag]:
        flags: list[ModalityFlag] = []
        if not keyframes:
            return flags
        # We require embeddings in lite mode. Prod mode would decode image_b64
        # via CLIPProcessor; left as TODO to keep this file CPU-only.
        for kf in keyframes:
            if kf.embedding is None:
                continue
            emb = np.asarray(kf.embedding, dtype=np.float64)
            n = np.linalg.norm(emb) or 1.0
            emb = emb / n
            for idx, (category, prompt, explanation) in enumerate(_ANCHORS):
                anchor = self._anchor_embed(idx, prompt, dim=emb.shape[0])
                sim = float(emb @ anchor)
                if sim < self.threshold:
                    continue
                flags.append(
                    ModalityFlag(
                        modality="visual",
                        category=category,
                        severity=min(1.0, max(0.0, (sim - self.threshold) /
                                              (1.0 - self.threshold))),
                        start_s=max(0.0, kf.t_s - 1.0),
                        end_s=kf.t_s + 1.0,
                        explanation=f"keyframe @ {kf.t_s:.1f}s: {explanation}",
                    )
                )
        return flags
