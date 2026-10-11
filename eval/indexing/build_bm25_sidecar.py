"""CLI: собрать BM25 sidecar из JSONL-экспорта Chroma.

Использование::

    python -m eval.indexing.build_bm25_sidecar \\
        --jsonl DATA/runs/corpus_full_export_20260512/chunks_from_chroma.jsonl \\
        --out-dir DATA/indexes/bm25/corpus_full_20260512 \\
        [--use-morph]

Запускается **один раз** на корпус (текст одинаковый для всех ``corpus_phase_b_*``
коллекций — отличается только эмбеддер). Результат используется как ``sparse``
ретривер в гибриде RRF/linear-weighted (F2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.rag.retrievers.sparse_bm25_retriever import build_bm25_index_from_jsonl


def main() -> int:
    p = argparse.ArgumentParser(description="Build BM25 sidecar index from corpus JSONL.")
    p.add_argument("--jsonl", required=True, type=Path, help="Path to chunks_from_chroma.jsonl")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory for BM25 sidecar")
    p.add_argument("--use-morph", action="store_true", help="Use pymorphy3 lemmatization (RU)")
    p.add_argument(
        "--text-field", default="document",
        help="JSONL field for chunk text (default: document)",
    )
    p.add_argument(
        "--id-field", default="id",
        help="JSONL field for chunk id (default: id, fallback chunk_id)",
    )
    args = p.parse_args()

    if not args.jsonl.is_file():
        print(f"jsonl not found: {args.jsonl}", file=sys.stderr)
        return 2

    meta = build_bm25_index_from_jsonl(
        jsonl_path=args.jsonl,
        out_dir=args.out_dir,
        text_field=args.text_field,
        id_field=args.id_field,
        use_morph=bool(args.use_morph),
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"OK: BM25 sidecar written to {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
