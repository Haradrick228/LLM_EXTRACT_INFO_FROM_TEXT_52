"""Агрегаты ранжирования поверх произвольной hit-функции.

Принимает любой ``hit_fn(doc_text, reference) -> bool`` (substring,
token-F1, semantic-similarity и т.д.) — это позволяет считать одни и те же
``recall@k / precision@k / MRR / nDCG@k`` при разных определениях релевантности.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, List

HitFn = Callable[[str, str], bool]


def _any_ref_hits(doc_text: str, references: Iterable[str], hit_fn: HitFn) -> bool:
    return any(hit_fn(doc_text, ref) for ref in references)


def recall_at_k(retrieved_texts: List[str], references: List[str], k: int, *, hit_fn: HitFn) -> float:
    if not references:
        return 0.0
    top = retrieved_texts[:k]
    matched = sum(1 for ref in references if any(hit_fn(d, ref) for d in top))
    return matched / max(1, len(references))


def precision_at_k(retrieved_texts: List[str], references: List[str], k: int, *, hit_fn: HitFn) -> float:
    if k <= 0:
        return 0.0
    top = retrieved_texts[:k]
    hits = sum(1 for d in top if _any_ref_hits(d, references, hit_fn))
    return hits / max(1, k)


def mean_reciprocal_rank(retrieved_texts: List[str], references: List[str], *, hit_fn: HitFn) -> float:
    for i, d in enumerate(retrieved_texts):
        if _any_ref_hits(d, references, hit_fn):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_texts: List[str], references: List[str], k: int, *, hit_fn: HitFn) -> float:
    """Бинарная релевантность, log2-discount, clipped at 1.0.

    ``nDCG`` может численно превышать 1.0, если одна и та же цитата матчится
    с несколькими top-документами (numerator растёт быстрее идеального
    знаменателя). Чтобы итоговый отчёт не путал, обрезаем сверху единицей —
    эта же логика была в legacy ``run_retrieval_eval``.
    """
    dcg = 0.0
    for i, d in enumerate(retrieved_texts[:k]):
        if _any_ref_hits(d, references, hit_fn):
            dcg += 1.0 if i == 0 else 1.0 / math.log2(i + 2)
    ideal_hits = min(len(references), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 if i == 0 else 1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg <= 0:
        return 0.0
    return min(1.0, dcg / idcg)


__all__ = ["recall_at_k", "precision_at_k", "mean_reciprocal_rank", "ndcg_at_k", "HitFn"]
