"""Deprecated re-export. См. :mod:`eval.indexing.export_chroma_collection_to_jsonl`."""
from __future__ import annotations

import warnings

from eval.indexing.export_chroma_collection_to_jsonl import *  # noqa: F401,F403

warnings.warn(
    "eval.export_chroma_run is deprecated; use eval.indexing.export_chroma_collection_to_jsonl",
    DeprecationWarning,
    stacklevel=2,
)
