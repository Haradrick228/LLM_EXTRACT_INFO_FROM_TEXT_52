"""Deprecated re-export. См. :mod:`eval.datasets.generate_synthetic_gold_v2`."""
from __future__ import annotations

import warnings

from eval.datasets.generate_synthetic_gold_v2 import *  # noqa: F401,F403

warnings.warn(
    "eval.generate_gold_v2_corpus is deprecated; use eval.datasets.generate_synthetic_gold_v2",
    DeprecationWarning,
    stacklevel=2,
)
