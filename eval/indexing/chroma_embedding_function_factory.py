"""Фабрика embedding_function для Chroma по полям из реестра баз (SentenceTransformers)."""
from __future__ import annotations

import os
from typing import Any, Dict

from chromadb.api.types import Documents, EmbeddingFunction


def _env_batch_size() -> int:
    return max(1, int(os.getenv("ST_EMBED_BATCH_SIZE", "32") or 32))


def _env_device() -> str:
    return os.getenv("ST_EMBED_DEVICE", os.getenv("RERANK_DEVICE", "cuda"))


def build_chroma_embedding_function(spec: Dict[str, Any]) -> EmbeddingFunction[Documents]:
    backend = str(spec.get("backend", "sentence_transformers")).strip().lower()
    if backend == "sentence_transformers":
        return _st_ef(spec)
    raise ValueError(
        f"Unknown backend: {backend!r}. Фаза B использует только sentence_transformers (HF через ST)."
    )


def _st_ef(spec: Dict[str, Any]) -> EmbeddingFunction[Documents]:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    model_name = str(spec.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("sentence_transformers base needs model_name")

    raw_kwargs = spec.get("st_model_kwargs") or {}
    if not isinstance(raw_kwargs, dict):
        raise ValueError("st_model_kwargs must be a dict")
    model_kwargs = {str(k): v for k, v in raw_kwargs.items()}
    # SentenceTransformer принимает trust_remote_code только на верхнем уровне; внутри model_kwargs
    # Chroma всё равно пробрасывает kwargs в ST, но ST игнорирует trust_remote_code внутри model_kwargs.
    trust_remote_code = bool(model_kwargs.pop("trust_remote_code", False))

    device = str(spec.get("st_device") or _env_device())
    normalize = bool(spec.get("st_normalize_embeddings", True))
    batch_size = int(spec.get("st_batch_size") or _env_batch_size())

    os.environ.setdefault("ST_EMBED_BATCH_SIZE", str(batch_size))

    st_kw: Dict[str, Any] = {
        "model_name": model_name,
        "device": device,
        "normalize_embeddings": normalize,
    }
    if trust_remote_code:
        st_kw["trust_remote_code"] = True
    if model_kwargs:
        st_kw["model_kwargs"] = model_kwargs
    return SentenceTransformerEmbeddingFunction(**st_kw)
