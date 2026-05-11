"""
Centralized reranker construction for RAG and eval scripts.

Keeps the same heuristics as the original RagService block: CrossEncoder when
`RERANK_MODE=cross` or id contains `cross-encoder`, otherwise bi-encoder SentenceTransformer.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore[misc, assignment]
    SentenceTransformer = None  # type: ignore[misc, assignment]


def load_reranker(
    model_name: str,
    mode: str,
    *,
    device: Optional[str] = None,
) -> Tuple[Optional[Any], str]:
    """
    Returns (model_or_none, kind) where kind is cross|bi|none.
    """
    name = (model_name or "").strip()
    if not name:
        return None, "none"

    mode_l = (mode or "auto").strip().lower()
    wants_cross = "cross-encoder" in name.lower() or mode_l == "cross"

    dev = (device if device is not None else os.getenv("RERANK_DEVICE") or "cuda").strip()
    ce_kw = {}
    st_kw = {}
    if dev and dev not in ("auto", "none"):
        ce_kw["device"] = dev
        st_kw["device"] = dev

    if wants_cross and CrossEncoder is not None:
        try:
            return CrossEncoder(name, **ce_kw), "cross"
        except TypeError:
            try:
                return CrossEncoder(name), "cross"
            except Exception:
                pass
        except Exception:
            pass

    if SentenceTransformer is None:
        return None, "none"
    try:
        return SentenceTransformer(name, **st_kw), "bi"
    except TypeError:
        try:
            return SentenceTransformer(name), "bi"
        except Exception:
            return None, "none"
    except Exception:
        return None, "none"
