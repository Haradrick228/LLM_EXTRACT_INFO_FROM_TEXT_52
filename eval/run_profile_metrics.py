"""Deprecated re-export. См. :mod:`eval.retrieval.run_profile_metrics`."""
from __future__ import annotations

import warnings

from eval.retrieval.run_profile_metrics import *  # noqa: F401,F403

warnings.warn(
    "eval.run_profile_metrics is deprecated; use eval.retrieval.run_profile_metrics",
    DeprecationWarning,
    stacklevel=2,
)
