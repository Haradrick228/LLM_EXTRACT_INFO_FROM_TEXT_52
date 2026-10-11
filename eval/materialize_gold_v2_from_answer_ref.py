"""Deprecated re-export. См. :mod:`eval.datasets.materialize_gold_v2_from_answer_ref`."""
from __future__ import annotations

import warnings

from eval.datasets.materialize_gold_v2_from_answer_ref import *  # noqa: F401,F403

warnings.warn(
    "eval.materialize_gold_v2_from_answer_ref is deprecated; "
    "use eval.datasets.materialize_gold_v2_from_answer_ref",
    DeprecationWarning,
    stacklevel=2,
)
