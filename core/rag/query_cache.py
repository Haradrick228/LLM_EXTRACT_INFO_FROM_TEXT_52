"""Deprecated re-export.

Старое местоположение кэша ответов LLM. Переехало в
:mod:`core.rag.service.llm_response_query_cache`.
"""
from __future__ import annotations

import warnings

from core.rag.service.llm_response_query_cache import *  # noqa: F401,F403
from core.rag.service.llm_response_query_cache import QueryCache  # noqa: F401

warnings.warn(
    "core.rag.query_cache is deprecated; use core.rag.service.llm_response_query_cache",
    DeprecationWarning,
    stacklevel=2,
)
