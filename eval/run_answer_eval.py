"""Deprecated re-export. См. :mod:`eval.retrieval.run_answer_quality_eval`."""
from __future__ import annotations

import warnings

from eval.retrieval.run_answer_quality_eval import *  # noqa: F401,F403

warnings.warn(
    "eval.run_answer_eval is deprecated; use eval.retrieval.run_answer_quality_eval",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    from eval.retrieval.run_answer_quality_eval import main

    main()
