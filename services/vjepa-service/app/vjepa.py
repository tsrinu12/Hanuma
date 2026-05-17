"""V-JEPA 2 wrapper: video -> embedding, video -> action labels.

V-JEPA (Video Joint-Embedding Predictive Architecture) is Meta's self-supervised
video representation model. V-JEPA 2 ships an attentive-probe classification head
fine-tuned on Something-Something V2 (174 action classes). For retrieval we use the
frozen encoder; for action recognition we use the classifier variant.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu, gpu  # type: ignore
from transformers import AutoModel, AutoVideoProcessor

try:
    # Available in transformers >=4.46 with V-JEPA 2 support
    from transformers import AutoModelForVideoClassification
except ImportError:  # pragma: no cover - keeps import safe on older versions
    AutoModelForVideoClassification = None  # type: ignore

from .config import settings

log = logging.getLogger("vjepa")


class VJEPAEngine:
    def __init__(self) -> None:
        self.device = torch.device(
            settings.device if torch.cuda.is_available() or settings.device == "cpu" else "cpu"
        )
        log.info("Loading V-JEPA encoder %s on %s", settings.vjepa_model, self.device)
        self.processor = AutoVideoProcessor.from_pretrained(settings.vjepa_model)
        self.encoder = AutoModel.from_pretrained(
            settings.vjepa_model, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device).eval()

        self.classifier = None
        if AutoModelForVideoClassification is not None:
            try:
                log.info("Loading V-JEPA classifier %s", settings.vjepa_classifier_model)
                self.classifier = AutoModelForVideoClassification.from_pretrained(
                    settings.vjepa_classifier_model,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                ).to(self.device).eval()
            except Exception as e:  # pragma: no cover
                log.warning("Classifier head unavailable: %s", e)

    # ---- frame sampling --------------------------------------------------

    def _sample_frames(self, video_path: str) -> np.ndarray:
        """Uniformly sample `frames_per_clip` frames from the video."""
        ctx = gpu(0) if self.device.type == "cuda" else cpu(0)
        try:
            vr = VideoReader(video_path, ctx=ctx)
        except Exception:
            vr = VideoReader(video_path, ctx=cpu(0))
        n = len(vr)
        idx = np.linspace(0, max(n - 1, 0), num=settings.frames_per_clip, dtype=int)
        frames = vr.get_batch(idx).asnumpy()  # (T, H, W, C) uint8
        return frames

    # ---- public API ------------------------------------------------------

    @torch.inference_mode()
    def embed(self, video_path: str) -> List[float]:
        frames = self._sample_frames(video_path)
        inputs = self.processor(videos=list(frames), return_tensors="pt").to(self.device)
        out = self.encoder(**inputs)
        # Mean-pool token embeddings to a single video-level vector
        last = out.last_hidden_state  # (B, N, D)
        vec = last.mean(dim=1).squeeze(0).float().cpu().numpy()
        # L2 normalize for cosine retrieval
        norm = np.linalg.norm(vec) + 1e-9
        return (vec / norm).tolist()

    @torch.inference_mode()
    def classify(self, video_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.classifier is None:
            raise RuntimeError("Classifier head not loaded — set VJEPA_CLASSIFIER_MODEL")
        frames = self._sample_frames(video_path)
        inputs = self.processor(videos=list(frames), return_tensors="pt").to(self.device)
        logits = self.classifier(**inputs).logits.squeeze(0).float().cpu()
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        id2label = self.classifier.config.id2label
        return [(id2label[int(i)], float(p)) for p, i in zip(topk.values, topk.indices)]
