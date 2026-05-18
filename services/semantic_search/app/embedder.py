"""Text embedder with two profiles.

In `MODEL_PROFILE=prod`, we use Sentence-Transformers BGE (small or large)
and produce both:
  - one pooled vector per chapter (for fast first-pass retrieval)
  - one per-token vector per chapter (for ColBERT-style late interaction)

In `MODEL_PROFILE=lite`, we use a deterministic hash-based embedder so the
service boots without downloading anything. The hash embedder preserves
substring overlap as a similarity signal, which is enough for the *plumbing*
to be tested end-to-end. Swap in the real embedder for production search.

The embedder is loaded lazily on first call so import time stays low and the
service can boot before model weights are present.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Protocol

import numpy as np

from .config import Settings


logger = logging.getLogger(__name__)


class Embedder(Protocol):
    @property
    def pooled_dim(self) -> int: ...
    @property
    def token_dim(self) -> int: ...
    def embed_pooled(self, texts: list[str]) -> np.ndarray: ...
    def embed_tokens(self, text: str, max_tokens: int) -> np.ndarray: ...


# ----- Lite (hash-based) embedder -----


_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9']+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _hash_vec(token: str, dim: int, seed: int = 0) -> np.ndarray:
    """Deterministic signed-hash vector for a token."""
    out = np.zeros(dim, dtype=np.float64)
    h = hashlib.blake2b(f"{seed}:{token}".encode(), digest_size=32).digest()
    # Spread 32 bytes across `dim` buckets.
    for i in range(min(dim, 32)):
        bucket = i % dim
        out[bucket] += float((h[i] & 0x7F) - 64) / 64.0
        if h[i] & 0x80:
            out[bucket] *= -1
    n = float(np.linalg.norm(out))
    return out if n == 0.0 else out / n


class HashEmbedder:
    """Lite embedder: deterministic, no downloads, fast.

    Pooled embedding = mean of token vectors, L2-normalized.
    Token embeddings = per-token vectors (good enough to exercise the
    late-interaction code path during tests).
    """

    def __init__(self, dim: int = 128) -> None:
        self._dim = dim

    @property
    def pooled_dim(self) -> int:
        return self._dim

    @property
    def token_dim(self) -> int:
        return self._dim

    def embed_pooled(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float64)
        for i, t in enumerate(texts):
            toks = _tokenize(t)
            if not toks:
                continue
            vecs = np.stack([_hash_vec(tok, self._dim) for tok in toks])
            v = vecs.mean(axis=0)
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            out[i] = v
        return out

    def embed_tokens(self, text: str, max_tokens: int) -> np.ndarray:
        toks = _tokenize(text)[:max_tokens]
        if not toks:
            return np.zeros((1, self._dim), dtype=np.float64)
        return np.stack([_hash_vec(tok, self._dim) for tok in toks])


# ----- Prod (transformer) embedder, lazy-loaded -----


class TransformerEmbedder:
    """Wraps a Sentence-Transformers / HuggingFace model.

    We load lazily because torch + transformers add 10s+ to import time.
    Only used when ``settings.is_prod`` is true.
    """

    def __init__(self, model_name: str, token_dim: int) -> None:
        self._model_name = model_name
        self._token_dim = token_dim
        self._model = None
        self._tok = None
        self._proj: np.ndarray | None = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "transformers not installed; install with extras "
                    "[prod] or set MODEL_PROFILE=lite"
                ) from e
            logger.info("loading transformer embedder: %s", self._model_name)
            self._tok = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)
            self._model.eval()
            # Random projection from hidden size to token_dim, fixed seed for
            # reproducibility.
            rng = np.random.default_rng(0)
            hidden_size = int(self._model.config.hidden_size)
            self._proj = rng.standard_normal((hidden_size, self._token_dim))
            self._proj /= np.linalg.norm(self._proj, axis=0, keepdims=True) + 1e-12

    @property
    def pooled_dim(self) -> int:
        self._load()
        return int(self._model.config.hidden_size)  # type: ignore[union-attr]

    @property
    def token_dim(self) -> int:
        return self._token_dim

    def embed_pooled(self, texts: list[str]) -> np.ndarray:
        import torch  # type: ignore

        self._load()
        with torch.no_grad():
            tok = self._tok(texts, padding=True, truncation=True, return_tensors="pt")  # type: ignore
            out = self._model(**tok)  # type: ignore
            # CLS-pool for BGE.
            v = out.last_hidden_state[:, 0, :].cpu().numpy()
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
            return v

    def embed_tokens(self, text: str, max_tokens: int) -> np.ndarray:
        import torch  # type: ignore

        self._load()
        with torch.no_grad():
            tok = self._tok(  # type: ignore
                text,
                padding=False,
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            )
            out = self._model(**tok)  # type: ignore
            v = out.last_hidden_state[0].cpu().numpy()  # (n_tokens, hidden)
            v = v @ self._proj
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
            return v


def build_embedder(settings: Settings) -> Embedder:
    if settings.is_prod:
        return TransformerEmbedder(
            model_name=settings.embedder_prod, token_dim=settings.token_embed_dim
        )
    return HashEmbedder(dim=settings.token_embed_dim)
