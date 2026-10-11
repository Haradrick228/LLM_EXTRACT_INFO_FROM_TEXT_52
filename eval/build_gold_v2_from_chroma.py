"""Deprecated re-export. См. :mod:`eval.datasets.build_gold_v2_from_chroma_chunks`."""
from __future__ import annotations

import warnings

from eval.datasets.build_gold_v2_from_chroma_chunks import *  # noqa: F401,F403

warnings.warn(
    "eval.build_gold_v2_from_chroma is deprecated; "
    "use eval.datasets.build_gold_v2_from_chroma_chunks",
    DeprecationWarning,
    stacklevel=2,
)
