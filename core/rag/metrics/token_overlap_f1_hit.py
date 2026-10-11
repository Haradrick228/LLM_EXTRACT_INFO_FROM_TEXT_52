"""Token-overlap F1 hit функция.

В отличие от substring-match, релевантность считается через F1 по
**нормализованным токенам**: lower-case + удаление пунктуации + бин-токены.
Порог настраивается (по умолчанию ``0.5``) — выше = строже.

Зачем такая метрика:

- substring fails, когда чанковщик режет цитату пополам;
- token-F1 терпим к перестановкам и переносам, при этом не «склеивает»
  далекие чанки;
- считается дёшево, не требует моделей.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.casefold() for t in _TOKEN_RE.findall(text or "")]


def token_overlap_f1(document_text: str, reference: str) -> float:
    """Микро-F1 по пересечению мультиmножеств токенов."""
    a = Counter(_tokenize(document_text))
    b = Counter(_tokenize(reference))
    if not a or not b:
        return 0.0
    overlap = sum((a & b).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(a.values())
    recall = overlap / sum(b.values())
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def make_token_overlap_f1_hit(threshold: float = 0.5):
    """Возвращает hit-функцию (doc, ref) -> bool с заданным порогом F1."""

    def _hit(document_text: str, reference: str) -> bool:
        return token_overlap_f1(document_text, reference) >= threshold

    _hit.__name__ = f"token_overlap_f1_hit_{int(threshold * 100)}"
    return _hit


def token_overlap_f1_hit(document_text: str, reference: str, *, threshold: float = 0.5) -> bool:
    """Default-обёртка с порогом ``0.5`` (используется как hit-fn в агрегатах)."""
    return token_overlap_f1(document_text, reference) >= threshold


__all__ = ["token_overlap_f1", "token_overlap_f1_hit", "make_token_overlap_f1_hit"]
