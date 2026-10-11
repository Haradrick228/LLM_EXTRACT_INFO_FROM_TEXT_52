"""Потоковое чтение JSONL чанков (id, document, metadata) для полнотекстовой индексации в Chroma."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from eval.text_pipeline import clean_text


def count_nonempty_lines(path: Path) -> int:
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def chroma_safe_metadata(meta: Any) -> Dict[str, Any]:
    """Chroma принимает только скаляры в metadata; остальное — JSON-строкой (с усечением)."""
    if not isinstance(meta, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in meta.items():
        key = str(k)
        if v is None:
            out[key] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[key] = v
        else:
            s = json.dumps(v, ensure_ascii=False)
            out[key] = s if len(s) <= 2000 else s[:1997] + "..."
    return out


def iter_jsonl_chunk_rows(
    path: Path,
    *,
    apply_clean_text: bool = True,
    max_rows: Optional[int] = None,
    skip_first_nonempty: int = 0,
) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    if skip_first_nonempty < 0:
        raise ValueError("skip_first_nonempty must be >= 0")
    skipped_nonempty = 0
    yielded = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if skipped_nonempty < skip_first_nonempty:
                skipped_nonempty += 1
                continue
            if max_rows is not None and yielded >= max_rows:
                break
            row = json.loads(line)
            doc_id = str(row["id"])
            raw = str(row.get("document", ""))
            text = clean_text(raw) if apply_clean_text else raw
            meta = chroma_safe_metadata(row.get("metadata") or {})
            # Chroma query() cannot request `ids` in `include` on current client; keep id in metadata for traces.
            meta["chunk_document_id"] = doc_id
            yield doc_id, text, meta
            yielded += 1


def iter_jsonl_batches(
    path: Path,
    batch_size: int,
    *,
    apply_clean_text: bool = True,
    max_rows: Optional[int] = None,
    skip_first_nonempty: int = 0,
) -> Iterator[Tuple[List[str], List[str], List[Dict[str, Any]]]]:
    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    for doc_id, text, meta in iter_jsonl_chunk_rows(
        path,
        apply_clean_text=apply_clean_text,
        max_rows=max_rows,
        skip_first_nonempty=skip_first_nonempty,
    ):
        ids.append(doc_id)
        docs.append(text)
        metas.append(meta)
        if len(ids) >= batch_size:
            yield ids, docs, metas
            ids, docs, metas = [], [], []
    if ids:
        yield ids, docs, metas
