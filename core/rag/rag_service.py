"""Deprecated re-export.

Старое местоположение ``RagService``. Переехало в
:mod:`core.rag.service.rag_service`. Этот shim остаётся ровно для обратной
совместимости импортов. Новый код должен импортировать напрямую из
``core.rag.service``.
"""
from __future__ import annotations

import warnings

from core.rag.service.rag_service import *  # noqa: F401,F403
from core.rag.service.rag_service import RagService  # noqa: F401

warnings.warn(
    "core.rag.rag_service is deprecated; use core.rag.service.rag_service",
    DeprecationWarning,
    stacklevel=2,
)
