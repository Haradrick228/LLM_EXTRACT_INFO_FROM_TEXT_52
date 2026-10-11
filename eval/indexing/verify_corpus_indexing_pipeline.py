"""
Проверки полного корпусного пайплайна (Chroma → JSONL → N коллекций с разными эмбеддерами).

Примеры:
  python -m eval.phase_b.verify_corpus_pipeline source-count --collection default_clean --embed-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
  python -m eval.phase_b.verify_corpus_pipeline jsonl-lines --jsonl DATA/runs/<id>/chunks_from_chroma.jsonl
  python -m eval.phase_b.verify_corpus_pipeline export-align --jsonl ... --manifest ... --source-count 62341
  python -m eval.phase_b.verify_corpus_pipeline index-align --expected-rows 62341
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.chroma_utils import chroma_http_client, get_collection  # noqa: E402
from eval.indexing.chroma_embedding_function_factory import build_chroma_embedding_function  # noqa: E402
from eval.indexing.jsonl_corpus_stream import count_nonempty_lines  # noqa: E402
from eval.experiments.phase_b_collections_registry import load_registry, list_bases  # noqa: E402

_DEFAULT_FULL_REGISTRY = _ROOT / "eval" / "config" / "embedding_bases_full_corpus.yaml"
if not _DEFAULT_FULL_REGISTRY.exists():
    _DEFAULT_FULL_REGISTRY = _ROOT / "eval" / "phase_b" / "embedding_bases_full_corpus.yaml"


def _resolve_chroma_collection(
    names: set[str],
    primary: str,
    fallback: str | None,
) -> str:
    """Как в run_hybrid_rrf_sweep: primary или collection_fallback."""
    if primary in names:
        return primary
    if fallback and fallback in names:
        print(
            f"[verify] chroma_collection={primary!r} missing; using collection_fallback={fallback!r}",
            file=sys.stderr,
        )
        return fallback
    raise ValueError(
        f"Chroma has neither collection={primary!r} nor fallback={fallback!r}. "
        f"Existing (sample): {sorted(list(names))[:30]}"
    )


def cmd_source_count(args: argparse.Namespace) -> int:
    client = chroma_http_client()
    col = get_collection(client, args.collection, args.embed_model)
    n = col.count()
    print(f"source_collection={args.collection!r} embed_model={args.embed_model!r} chroma_count={n}")
    return 0


def cmd_jsonl_lines(args: argparse.Namespace) -> int:
    p = Path(args.jsonl).resolve()
    if not p.is_file():
        print("ERR: jsonl not found:", p, file=sys.stderr)
        return 2
    n = count_nonempty_lines(p)
    print(f"jsonl={p} nonempty_lines={n}")
    return 0


def cmd_export_align(args: argparse.Namespace) -> int:
    jp = Path(args.jsonl).resolve()
    if not jp.is_file():
        print("ERR: jsonl not found:", jp, file=sys.stderr)
        return 2
    lines = count_nonempty_lines(jp)
    manifest_rows: int | None = None
    if args.manifest:
        mp = Path(args.manifest).resolve()
        if not mp.is_file():
            print("ERR: manifest not found:", mp, file=sys.stderr)
            return 2
        man: Dict[str, Any] = json.loads(mp.read_text(encoding="utf-8"))
        manifest_rows = int(man.get("rows_exported", -1))
        if manifest_rows < 0:
            print("ERR: manifest missing rows_exported", file=sys.stderr)
            return 2

    src = int(args.source_count)
    ok = lines == src
    print(f"source_count={src} jsonl_nonempty_lines={lines} match={ok}")
    if manifest_rows is not None:
        ok_m = manifest_rows == lines == src
        print(f"manifest_rows_exported={manifest_rows} triple_match={ok_m}")
        return 0 if ok_m else 1
    return 0 if ok else 1


def cmd_index_align(args: argparse.Namespace) -> int:
    reg = Path(args.registry).resolve() if args.registry else _DEFAULT_FULL_REGISTRY
    data = load_registry(reg)
    expected = int(args.expected_rows)
    client = chroma_http_client()
    chroma_names = {c.name for c in client.list_collections()}
    only_ids: set[str] | None = None
    if getattr(args, "base_ids", "").strip():
        only_ids = {x.strip() for x in str(args.base_ids).split(",") if x.strip()}
    ok = True
    print(f"expected_rows={expected} registry={reg}")
    if only_ids:
        print(f"base_ids_filter={sorted(only_ids)}")
    print("base_id\tcollection\tcount\tok")
    for b in list_bases(data):
        bid = str(b.get("base_id", ""))
        if only_ids is not None and bid not in only_ids:
            continue
        merged = dict(data.get("defaults") or {})
        merged.update(b)
        primary = str(merged["collection"])
        fb = merged.get("collection_fallback")
        fallback = str(fb).strip() if fb else None
        try:
            name = _resolve_chroma_collection(chroma_names, primary, fallback)
        except ValueError as exc:
            print(f"{bid}\t{primary}\t(missing)\tFalse\t{exc}", file=sys.stderr)
            ok = False
            continue
        ef = build_chroma_embedding_function(merged)
        col = client.get_collection(name, embedding_function=ef)
        n = col.count()
        row_ok = n == expected
        ok = ok and row_ok
        print(f"{bid}\t{name}\t{n}\t{row_ok}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify full-corpus Chroma export + multi-embedder index.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("source-count", help="Chroma count() для исходной коллекции.")
    p1.add_argument("--collection", required=True)
    p1.add_argument(
        "--embed-model",
        required=True,
        help="Тот же model_name, с которым коллекция создавалась в Chroma.",
    )
    p1.set_defaults(func=cmd_source_count)

    p2 = sub.add_parser("jsonl-lines", help="Число непустых строк в JSONL.")
    p2.add_argument("--jsonl", required=True)
    p2.set_defaults(func=cmd_jsonl_lines)

    p3 = sub.add_parser(
        "export-align",
        help="Сверка: source_count == jsonl lines == manifest.rows_exported (если --manifest).",
    )
    p3.add_argument("--jsonl", required=True)
    p3.add_argument("--source-count", type=int, required=True)
    p3.add_argument("--manifest", default="", help="manifest.json от export_chroma_run.")
    p3.set_defaults(func=cmd_export_align)

    p4 = sub.add_parser("index-align", help="Все коллекции из full-corpus registry имеют count == expected_rows.")
    p4.add_argument("--expected-rows", type=int, required=True)
    p4.add_argument("--registry", type=Path, default=None)
    p4.add_argument(
        "--base-ids",
        default="",
        help="Через запятую: только эти base_id (например шесть рабочих без Giga/Qwen).",
    )
    p4.set_defaults(func=cmd_index_align)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
