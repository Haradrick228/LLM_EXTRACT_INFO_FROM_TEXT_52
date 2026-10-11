"""Deprecated re-export. См. :mod:`eval.datasets.validate_gold_v2_schema`."""
from __future__ import annotations

import warnings

from eval.datasets.validate_gold_v2_schema import *  # noqa: F401,F403

warnings.warn(
    "eval.validate_gold_v2 is deprecated; use eval.datasets.validate_gold_v2_schema",
    DeprecationWarning,
    stacklevel=2,
)
