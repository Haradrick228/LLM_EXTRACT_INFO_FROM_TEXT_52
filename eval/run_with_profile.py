"""Deprecated re-export. См. :mod:`eval.retrieval.run_eval_with_profile`."""
from __future__ import annotations

import warnings

from eval.retrieval.run_eval_with_profile import *  # noqa: F401,F403

warnings.warn(
    "eval.run_with_profile is deprecated; use eval.retrieval.run_eval_with_profile",
    DeprecationWarning,
    stacklevel=2,
)
