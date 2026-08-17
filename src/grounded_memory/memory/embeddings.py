"""Embedding backends for memory events.

The default ``HashingTextEmbedder`` is deterministic and dependency-free (hashing
vectorizer + L2 normalize) so the memory/retrieval stack and its tests run with
no model downloads. Swap in a real sentence/image encoder for actual experiments
via the ``Embedder`` protocol.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed_text(self, text: str) -> np.ndarray: ...


class HashingTextEmbedder:
    """Deterministic bag-of-tokens hashing embedder (no external models).

    Good enough to exercise retrieval logic and tests; NOT a semantic model.
    For real semantic retrieval, replace with a sentence-transformer encoder.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _token_index(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()  # noqa: S324 (non-crypto use)
        return int(h, 16) % self.dim

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _tokenize(text):
            vec[self._token_index(token)] += 1.0
        return l2_normalize(vec)


class OllamaEmbedder:
    """Real semantic embeddings from a local Ollama embedding model.

    Defaults to ``nomic-embed-text`` (768-d). Use this to replace the
    ``HashingTextEmbedder`` stand-in so E2 semantic/hybrid retrieval is genuinely
    semantic. Requires the Ollama server running and the model pulled
    (``ollama pull nomic-embed-text``). Stdlib HTTP only — no extra deps.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        endpoint: str | None = None,
        dim: int = 768,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        from grounded_memory import ollama_client

        resp = ollama_client.post_json(
            "/api/embeddings", {"model": self.model, "prompt": text}, endpoint=self.endpoint
        )
        vec = np.asarray(resp["embedding"], dtype=np.float32)
        self.dim = int(vec.shape[0])
        return l2_normalize(vec)


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / (norm + eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


__all__ = [
    "Embedder",
    "HashingTextEmbedder",
    "OllamaEmbedder",
    "l2_normalize",
    "cosine_similarity",
]
