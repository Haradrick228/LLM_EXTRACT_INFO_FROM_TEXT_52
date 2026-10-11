"""Пакет метрик retrieval.

Каждая «hit-функция» определяет, считается ли документ релевантным
относительно ``references[i]``. ``ranking_aggregates`` поверх hit-функции
считает ``recall@k / precision@k / MRR / nDCG@k``.

Реализации:

- :mod:`core.rag.metrics.substring_match_hit` — substring (как в README/JSON);
- :mod:`core.rag.metrics.token_overlap_f1_hit` — F1 по нормализованным токенам;
- :mod:`core.rag.metrics.semantic_similarity_hit` — cosine между ref и chunk;
- :mod:`core.rag.metrics.source_id_match_hit` — hit по совпадению source/chunk_id;
- :mod:`core.rag.metrics.ranking_aggregates` — recall@k / precision@k / MRR / nDCG@k.
"""
from .ranking_aggregates import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .semantic_similarity_hit import (
    SemanticSimilarityScorer,
    build_semantic_similarity_scorer,
)
from .source_id_match_hit import source_id_match_hit
from .substring_match_hit import substring_match_hit
from .token_overlap_f1_hit import (
    make_token_overlap_f1_hit,
    token_overlap_f1,
    token_overlap_f1_hit,
)

__all__ = [
    "substring_match_hit",
    "token_overlap_f1",
    "token_overlap_f1_hit",
    "make_token_overlap_f1_hit",
    "build_semantic_similarity_scorer",
    "SemanticSimilarityScorer",
    "source_id_match_hit",
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
]
