"""Deprecated re-export.

Старое местоположение фабрики реранкера. Переехало в
:mod:`core.rag.rerankers.cross_encoder_reranker`.

Этот модуль остаётся как shim ровно для обратной совместимости импортов в
существующем коде; новый код должен импортировать напрямую из
``core.rag.rerankers``.
"""
from __future__ import annotations

import warnings

from core.rag.rerankers.cross_encoder_reranker import (  # noqa: F401
    CrossEncoderReranker,
    build_reranker,
    load_reranker,
)

warnings.warn(
    "core.rag.rerank_registry is deprecated; use core.rag.rerankers.cross_encoder_reranker",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["CrossEncoderReranker", "build_reranker", "load_reranker"]
