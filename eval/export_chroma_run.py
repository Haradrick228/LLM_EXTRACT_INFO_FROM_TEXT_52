"""Export a Chroma collection to DATA/runs/<run_id>/ with manifest (append-only research artifact)."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

from common.rag_constants import CHUNK_OVERLAP, CHUNK_SIZE

# repo root
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Chroma collection documents to DATA/runs/<run_id>/")
    parser.add_argument("--run-id", default="", help="Directory name under DATA/runs (default: chroma_export_<unix>)")
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "default_clean"))
    parser.add_argument("--embed-model", default=os.getenv("CHROMA_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
    args = parser.parse_args()

    run_id = args.run_id.strip() or f"chroma_export_{int(time.time())}"
    out_dir = ROOT / "DATA" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    from eval.chroma_utils import chroma_http_client, export_collection_to_jsonl, get_collection

    client = chroma_http_client()
    col = get_collection(client, args.collection, args.embed_model)
    out_jsonl = out_dir / "chunks_from_chroma.jsonl"
    n = export_collection_to_jsonl(col, out_jsonl)

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "created_unix": int(time.time()),
        "source": "chroma_export",
        "collection": args.collection,
        "embed_model": args.embed_model,
        "chunk_size_default": CHUNK_SIZE,
        "chunk_overlap_default": CHUNK_OVERLAP,
        "rows_exported": n,
        "output_jsonl": str(out_jsonl.relative_to(ROOT)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Exported {n} rows to {out_jsonl}")


if __name__ == "__main__":
    main()
