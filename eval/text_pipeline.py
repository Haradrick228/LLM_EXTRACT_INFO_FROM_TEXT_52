"""Clean and chunk text (logic aligned with analysis.eda.text.TextProcessor; standalone to avoid heavy eda imports)."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

from common.rag_constants import CHUNK_OVERLAP, CHUNK_SIZE


def clean_text(text: str) -> str:
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\ufeff", " ")
    normalized = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]", " ", normalized)
    normalized = re.sub(r"[^\S\n]+", " ", normalized)
    normalized = re.sub(r"\s+\n", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    normalized = re.sub(r"[ \t]+$", "", normalized, flags=re.MULTILINE)
    return normalized.strip()


def chunk_text(text: str) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    end = len(text)
    while start < end:
        finish = min(start + CHUNK_SIZE, end)
        chunks.append(text[start:finish])
        if finish == end:
            break
        start = max(finish - CHUNK_OVERLAP, start + 1)
    return chunks


def write_chunks_jsonl(chunks: List[str], out_path: Path, *, source: str) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, t in enumerate(chunks):
            row: Dict[str, Any] = {
                "id": f"{source}::{i}",
                "document": t,
                "metadata": {"source": source, "chunk_index": i},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
