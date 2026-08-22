"""
Eval-loop adapter: app.embedder
================================
Satisfies the rag-local-eval-loop TARGET_INTERFACE.md contract for the
embedder module. Wraps the FastEmbed TextEmbedding model already used by
HybridRetriever so the eval loop measures the *real* embedding stack, not
a stub.

Required surface (TARGET_INTERFACE.md):
  get_model()          -- load/return the model (called once; side-effect matters)
  embed(texts) -> ndarray (N, dim)
  embed_one(text) -> ndarray (dim,)   -- must support .reshape(1,-1) and .shape[-1]
"""
from __future__ import annotations

import os
import numpy as np

# Respect the same thread-pinning the production container uses
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ONNX_NUM_THREADS", "1")

from .config import settings

_model = None
_MODEL_NAME = settings().embedding_model  # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_model():
    """Load (or return cached) FastEmbed TextEmbedding model."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        threads = int(os.getenv("EMBEDDING_THREADS", "1"))
        _model = TextEmbedding(model_name=_MODEL_NAME, threads=threads)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts, returning shape (len(texts), dim)."""
    if not texts:
        # Discover dim from a dummy call
        dummy = embed_one("hello")
        return np.zeros((0, dummy.shape[-1]), dtype=np.float32)
    model = get_model()
    vecs = list(model.embed(texts))
    return np.asarray(vecs, dtype=np.float32)


def embed_one(text: str) -> np.ndarray:
    """Embed a single text, returning shape (dim,).
    Supports .reshape(1, -1) and .shape[-1] as required by the eval loop.
    """
    model = get_model()
    # Use query_embed for asymmetric retrieval (same as production path)
    vec = next(model.query_embed([text]))
    return np.asarray(vec, dtype=np.float32)
