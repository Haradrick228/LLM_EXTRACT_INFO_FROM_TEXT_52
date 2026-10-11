"""Substring-match hit функция.

Документ считается релевантным относительно ``reference``, если одна строка
является подстрокой другой после ``lower()``. Это та же логика, которая
исторически использовалась в ``eval/run_retrieval_eval.py`` для recall@k /
precision@k / MRR / nDCG@k.

Метрика грубая — длинные ``references``, которые чанкование разрезало
пополам, не матчатся ни с одной из половинок. Поэтому в фазе F1 рядом
живут более точные ``token_overlap_f1_hit`` и ``semantic_similarity_hit``.
"""
from __future__ import annotations


def substring_match_hit(document_text: str, reference: str) -> bool:
    """Возвращает True, если ``reference`` ∈ ``document_text`` или наоборот."""
    if not document_text or not reference:
        return False
    a = document_text.lower()
    b = reference.lower()
    return (b in a) or (a in b)


__all__ = ["substring_match_hit"]
