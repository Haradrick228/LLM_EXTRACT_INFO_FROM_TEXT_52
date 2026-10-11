"""
Single stderr banner for eval scripts so logs show the full component combo
(embedder, reranker, collection, chunking hint, LLM, etc.).
"""
from __future__ import annotations

import sys
from typing import Mapping, Optional


def write_eval_banner(component: str, fields: Mapping[str, object], *, note: Optional[str] = None) -> None:
    lines = [f"[eval_config] component={component}"]
    for key in sorted(fields.keys()):
        val = fields[key]
        if val is None or val == "":
            continue
        lines.append(f"  {key}={val!r}")
    if note:
        lines.append(f"  _note={note!r}")
    sys.stderr.write("\n".join(lines) + "\n")
    sys.stderr.flush()
