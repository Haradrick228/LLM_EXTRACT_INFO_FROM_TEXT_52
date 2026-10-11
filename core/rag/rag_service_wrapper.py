"""Deprecated re-export.

Старое местоположение «обёртки». Переехало в
:mod:`core.rag.service.rag_service_legacy_facade`.
"""
from __future__ import annotations

import warnings

from core.rag.service.rag_service_legacy_facade import *  # noqa: F401,F403

warnings.warn(
    "core.rag.rag_service_wrapper is deprecated; use core.rag.service.rag_service_legacy_facade",
    DeprecationWarning,
    stacklevel=2,
)
