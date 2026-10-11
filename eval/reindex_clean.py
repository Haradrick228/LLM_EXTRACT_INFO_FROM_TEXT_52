"""Deprecated re-export. См. :mod:`eval.indexing.reindex_with_text_cleaning`."""
from __future__ import annotations

import warnings

from eval.indexing.reindex_with_text_cleaning import *  # noqa: F401,F403

warnings.warn(
    "eval.reindex_clean is deprecated; use eval.indexing.reindex_with_text_cleaning",
    DeprecationWarning,
    stacklevel=2,
)
