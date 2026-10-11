"""Deprecated re-export.

CLI переехал в :mod:`eval.retrieval.run_retrieval_eval`. Запускайте как
``python -m eval.retrieval.run_retrieval_eval``.

Этот shim сохраняет работоспособность старых импортов в течение
переходного периода. Новый код должен импортировать напрямую из
``eval.retrieval.run_retrieval_eval``.
"""
from __future__ import annotations

import warnings

from eval.retrieval.run_retrieval_eval import *  # noqa: F401,F403
from eval.retrieval.run_retrieval_eval import main  # noqa: F401

warnings.warn(
    "eval.run_retrieval_eval is deprecated; use eval.retrieval.run_retrieval_eval",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    main()
