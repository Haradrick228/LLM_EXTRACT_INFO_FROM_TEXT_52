"""Deprecated re-export.

Старое местоположение клиента DeepSeek. Переехало в
:mod:`core.rag.llm.deepseek_llm_client`.
"""
from __future__ import annotations

import warnings

from core.rag.llm.deepseek_llm_client import *  # noqa: F401,F403

warnings.warn(
    "core.rag.deepseek_llm is deprecated; use core.rag.llm.deepseek_llm_client",
    DeprecationWarning,
    stacklevel=2,
)
