"""Semantic-similarity hit функция.

Cosine между эмбеддингом ``document_text`` и ``reference``. Считается
один раз для всей матрицы (lazy в фабрике), запросы к модели
батчуются — иначе в run_retrieval_eval перегрев CPU/GPU.

API:

- ``build_semantic_similarity_scorer(model)`` — фабрика, возвращает
  callable ``score(document_text, reference) -> float`` и hit-функцию ``hit(...)``
  с настраиваемым порогом.
- если ``sentence-transformers`` недоступен или модель не грузится — fallback
  на substring match, чтобы CLI продолжал работать.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from common.hf_model_resolution import resolve_hf_model_path
from .substring_match_hit import substring_match_hit


@dataclass(frozen=True)
class SemanticSimilarityScorer:
    """Контейнер с двумя callable'ами: score и hit."""

    score_fn: Callable[[str, str], float]
    hit_fn: Callable[[str, str], bool]
    model_name: str
    threshold: float
    backend: str  # "sentence-transformers" | "substring-fallback"


def _build_substring_fallback(threshold: float) -> SemanticSimilarityScorer:
    def _score(a: str, b: str) -> float:
        return 1.0 if substring_match_hit(a, b) else 0.0

    def _hit(a: str, b: str) -> bool:
        return substring_match_hit(a, b)

    return SemanticSimilarityScorer(
        score_fn=_score,
        hit_fn=_hit,
        model_name="substring-fallback",
        threshold=threshold,
        backend="substring-fallback",
    )


def build_semantic_similarity_scorer(
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    *,
    threshold: float = 0.62,
    device: Optional[str] = None,
) -> SemanticSimilarityScorer:
    """Создаёт scorer; при отсутствии sentence-transformers — substring fallback."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return _build_substring_fallback(threshold)

    try:
        kw: dict[str, Any] = {}
        if device:
            kw["device"] = device
        model = SentenceTransformer(resolve_hf_model_path(model_name), **kw)
    except Exception:
        return _build_substring_fallback(threshold)

    def _score(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        embs = model.encode([a, b], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return float((embs[0] * embs[1]).sum())

    def _hit(a: str, b: str) -> bool:
        return _score(a, b) >= threshold

    return SemanticSimilarityScorer(
        score_fn=_score,
        hit_fn=_hit,
        model_name=model_name,
        threshold=threshold,
        backend="sentence-transformers",
    )


__all__ = ["SemanticSimilarityScorer", "build_semantic_similarity_scorer"]
