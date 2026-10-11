"""Run ``run_answer_quality_eval`` using metadata from a retrieval run directory.

Reads ``retrieval.json`` ``_meta`` (if present) to align ``CHROMA_*``, ``ANSWER_EVAL_FETCH_K``,
and ``ANSWER_EVAL_FILE`` with the retrieval eval. Writes ``answer_eval_results.jsonl`` and
``answer_eval_manifest.json`` under the same ``--run-dir``.

Usage::

    python -m eval.diagnostics.run_answer_eval_in_run_dir \\
        --run-dir eval/runs/20260513_205851_gold_on_full_corpus_bge_m3 \\
        --max-rows 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.hf_model_resolution import embedder_needs_trust_remote_code


def _load_retrieval_meta(run_dir: Path, retrieval_json: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    if retrieval_json is not None:
        p = retrieval_json.resolve()
    else:
        p = run_dir / "retrieval.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    meta = data.get("_meta")
    return meta if isinstance(meta, dict) else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="Override gold JSONL (default: from retrieval.json _meta.eval_file).",
    )
    p.add_argument(
        "--retrieval-json",
        type=Path,
        default=None,
        help="Путь к retrieval_*.json (например .../full_bge_m3/retrieval_rerank_jina_v3.json). "
        "По умолчанию: <run-dir>/retrieval.json",
    )
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows in dataset.")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"run-dir not found: {run_dir}", file=sys.stderr)
        return 2

    meta = _load_retrieval_meta(run_dir, args.retrieval_json)
    eval_rel = None
    fetch_k = None
    embed_model = None
    collection = None
    host = os.getenv("RAG_EVAL_CHROMA_HOST", os.getenv("CHROMA_HOST", "127.0.0.1"))
    port = os.getenv("RAG_EVAL_CHROMA_PORT", os.getenv("CHROMA_PORT", "18000"))

    if meta:
        eval_rel = meta.get("eval_file")
        fetch_k = meta.get("fetch_k")
        embed_model = meta.get("embed_model")
        collection = meta.get("chroma_collection")
        host = str(meta.get("chroma_host") or host)
        port = str(meta.get("chroma_port") or port)
        rm = meta.get("rerank_model")
        rmode = meta.get("rerank_mode")
        if isinstance(rm, str) and rm.strip():
            os.environ["RERANK_MODEL"] = rm.strip()
        if isinstance(rmode, str) and rmode.strip():
            os.environ["RERANK_MODE"] = rmode.strip()

    eval_path = args.eval_file
    if eval_path is None:
        if not eval_rel:
            print("No --eval-file and no retrieval.json _meta.eval_file", file=sys.stderr)
            return 2
        eval_path = (_ROOT / str(eval_rel)).resolve() if not Path(str(eval_rel)).is_absolute() else Path(str(eval_rel))

    if not eval_path.is_file():
        print(f"eval file not found: {eval_path}", file=sys.stderr)
        return 2

    os.environ["ANSWER_EVAL_FILE"] = str(eval_path)
    os.environ["ANSWER_EVAL_OUT"] = str(run_dir / "answer_eval_results.jsonl")
    os.environ["CHROMA_HOST"] = str(host).strip()
    os.environ["CHROMA_PORT"] = str(port).strip()
    if collection:
        os.environ["CHROMA_COLLECTION"] = str(collection)
    if embed_model:
        os.environ["CHROMA_EMBED_MODEL"] = str(embed_model)
        if embedder_needs_trust_remote_code(str(embed_model)):
            os.environ["CHROMA_EMBED_TRUST_REMOTE_CODE"] = "1"
    if fetch_k is not None:
        fk = int(fetch_k)
        os.environ["ANSWER_EVAL_FETCH_K"] = str(fk)
        os.environ["RETRIEVER_FETCH_K"] = str(fk)
    os.environ.setdefault("RETRIEVER_K", os.getenv("RETRIEVER_K", "10"))
    os.environ["GRAPH_RAG_ENABLED"] = "0"
    os.environ["LLM_CACHE_ENABLED"] = "0"
    if args.max_rows > 0:
        os.environ["ANSWER_EVAL_MAX_ROWS"] = str(int(args.max_rows))

    scen = os.getenv("EVAL_SCENARIO_TYPE", "").strip() or "answer_slice_in_run_dir"
    os.environ["EVAL_SCENARIO_TYPE"] = f"{scen}::{run_dir.name}"

    manifest = {
        "schema": "answer_eval_in_run_dir_v1",
        "run_dir": str(run_dir.as_posix()),
        "answer_eval_file": str(eval_path.as_posix()),
        "answer_eval_out": os.environ["ANSWER_EVAL_OUT"],
        "fetch_k": os.environ.get("ANSWER_EVAL_FETCH_K"),
        "chroma_collection": os.environ.get("CHROMA_COLLECTION"),
        "chroma_embed_model": os.environ.get("CHROMA_EMBED_MODEL"),
        "max_rows": int(args.max_rows or 0),
        "rerank_model": os.environ.get("RERANK_MODEL"),
        "rerank_mode": os.environ.get("RERANK_MODE"),
        "retrieval_json": str(args.retrieval_json.as_posix()) if args.retrieval_json else None,
    }
    (run_dir / "answer_eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    from eval.retrieval.run_answer_quality_eval import main as answer_main

    answer_main()
    print("Manifest:", run_dir / "answer_eval_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
