"""Source-/chunk-id match: «правильный документ найден», независимо от чанкования.

Эта метрика ОДНОЗНАЧНА — нет никакой подстрочной/семантической оценки.
Используется когда gold-строка содержит ``source`` (имя исходного документа)
или ``chunk_id`` и retrieval должен вернуть документ ИМЕННО оттуда.

Применение:

- gold-строка может иметь ``source`` (например, ``edu_curriculum_2023.pdf``)
  или ``chunk_id`` (стабильный id чанка в Chroma);
- ретривер возвращает ``RetrievedDoc`` с ``source``/``chunk_id`` в metadata;
- считается, что ``hit = True`` если совпадает хотя бы один из этих ключей.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def source_id_match_hit(
    retrieved: Mapping[str, object],
    *,
    references_source: Iterable[Optional[str]] = (),
    references_chunk_id: Iterable[Optional[str]] = (),
) -> bool:
    """True если retrieved.source / retrieved.chunk_id матчат любую из refs."""
    src = _norm(str(retrieved.get("source", "")))
    cid = _norm(str(retrieved.get("chunk_id", "")))
    refs_src = {_norm(s) for s in references_source if s}
    refs_cid = {_norm(s) for s in references_chunk_id if s}
    if src and src in refs_src:
        return True
    if cid and cid in refs_cid:
        return True
    return False


__all__ = ["source_id_match_hit"]
