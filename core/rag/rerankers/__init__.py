"""Пакет реранкеров.

Конкретные реализации:

- :mod:`core.rag.rerankers.cross_encoder_reranker` — CrossEncoder / bi-encoder
  через ``sentence-transformers`` (исторический ``rerank_registry``);
- :mod:`core.rag.rerankers.colbert_late_interaction_reranker` — F4 baseline.

Фабрика по имени модели — ``build_reranker(name, mode, device)``.
"""
from .cross_encoder_reranker import build_reranker, load_reranker
from .reranker_base_types import Reranker, RerankRequest, RerankResult

__all__ = [
    "Reranker",
    "RerankRequest",
    "RerankResult",
    "build_reranker",
    "load_reranker",
]
