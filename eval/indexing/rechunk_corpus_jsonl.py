"""F5: пере-чанковка корпуса JSONL → новый JSONL.

Принимает на входе JSONL ``{id, document, metadata}`` (формат
``eval.indexing.export_chroma_collection_to_jsonl``), применяет выбранный
:class:`Chunker` к каждому ``document`` и пишет новый JSONL того же формата.

Используется как препроцесс перед ``eval.indexing.index_corpus_from_jsonl`` для
сравнения «flat vs parent-child vs semantic» chunking стратегий.

Использование::

    # Parent-Child (1024/256 → 256/64)
    python -m eval.indexing.rechunk_corpus_jsonl \\
        --in-jsonl DATA/runs/corpus_full_export_20260512/chunks_from_chroma.jsonl \\
        --out-jsonl DATA/runs/corpus_full_export_pc_1024_256/chunks.jsonl \\
        --chunker parent_child --parent-size 1024 --parent-overlap 128 \\
        --child-size 256 --child-overlap 64

    # Semantic (требует --embedder-model)
    python -m eval.indexing.rechunk_corpus_jsonl \\
        --in-jsonl ... --out-jsonl ... \\
        --chunker semantic --embedder-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2

JSONL: каждая строка содержит ``{id, document, metadata}`` — для совместимости
с :mod:`eval.indexing.index_corpus_from_jsonl`. У parent-child записываются
**child** чанки; ``metadata.parent_chunk_id`` и ``metadata.parent_text``
позволяют поднять родителя в при retrieval.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.rag.chunkers.chunker_base_types import SourceDocument  # noqa: E402
from core.rag.chunkers.fixed_size_chunker import FixedSizeChunker  # noqa: E402
from core.rag.chunkers.parent_child_chunker import ParentChildChunker  # noqa: E402
from core.rag.chunkers.semantic_boundary_chunker import SemanticBoundaryChunker  # noqa: E402


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as exc:
                raise ValueError(f"jsonl parse error at line {line_no}: {exc}") from exc


def _build_chunker(args: argparse.Namespace):
    kind = args.chunker.strip().lower()
    if kind == "fixed":
        return FixedSizeChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            name="fixed_size",
        )
    if kind == "parent_child":
        return ParentChildChunker(
            parent_size=args.parent_size,
            parent_overlap=args.parent_overlap,
            child_size=args.child_size,
            child_overlap=args.child_overlap,
            name="parent_child",
        )
    if kind == "semantic":
        if not args.embedder_model:
            raise SystemExit("--embedder-model is required for chunker=semantic")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dep
            raise SystemExit(
                f"sentence-transformers is required for chunker=semantic: {exc!r}"
            ) from exc
        model = SentenceTransformer(args.embedder_model, device=args.embedder_device or None)

        def _embed(sentences: List[str]) -> List[List[float]]:
            vecs = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
            return [list(map(float, v)) for v in vecs]

        return SemanticBoundaryChunker(
            embedder=_embed,
            similarity_threshold=args.similarity_threshold,
            breakpoint_percentile=args.breakpoint_percentile,
            max_chunk_tokens=args.max_chunk_tokens,
            min_sentences_per_chunk=args.min_sentences_per_chunk,
            name="semantic_boundary",
        )
    raise SystemExit(f"unknown chunker={kind!r}; choose fixed|parent_child|semantic")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-jsonl", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--chunker", choices=["fixed", "parent_child", "semantic"], required=True)
    p.add_argument("--text-field", default="document")
    p.add_argument("--id-field", default="id")
    p.add_argument("--limit", type=int, default=0, help="0 = all source documents")
    p.add_argument("--chunk-size", type=int, default=650)
    p.add_argument("--chunk-overlap", type=int, default=250)
    p.add_argument("--parent-size", type=int, default=1024)
    p.add_argument("--parent-overlap", type=int, default=128)
    p.add_argument("--child-size", type=int, default=256)
    p.add_argument("--child-overlap", type=int, default=64)
    p.add_argument("--embedder-model", default="")
    p.add_argument("--embedder-device", default="")
    p.add_argument("--similarity-threshold", type=float, default=0.55)
    p.add_argument("--breakpoint-percentile", type=float, default=0.2)
    p.add_argument("--max-chunk-tokens", type=int, default=768)
    p.add_argument("--min-sentences-per-chunk", type=int, default=1)
    args = p.parse_args()

    if not args.in_jsonl.is_file():
        print(f"input jsonl not found: {args.in_jsonl}", file=sys.stderr)
        return 2

    chunker = _build_chunker(args)
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_docs = 0
    n_chunks = 0
    t0 = time.monotonic()
    with args.out_jsonl.open("w", encoding="utf-8") as fout:
        for _ln, row in _iter_jsonl(args.in_jsonl):
            if args.limit and n_docs >= args.limit:
                break
            doc_id = str(row.get(args.id_field) or f"row_{n_docs}")
            text = str(row.get(args.text_field) or "")
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            src = SourceDocument(doc_id=doc_id, text=text, metadata=meta)
            result = chunker.split(src)

            # Для parent-child кладём родительский текст в metadata детей.
            parents_by_id: Dict[str, str] = {p.chunk_id: p.text for p in result.parents}
            for ch in result.chunks:
                md: Dict[str, Any] = dict(ch.metadata)
                if ch.parent_id and ch.parent_id in parents_by_id:
                    md["parent_text"] = parents_by_id[ch.parent_id]
                out_row = {
                    "id": ch.chunk_id,
                    "document": ch.text,
                    "metadata": md,
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                n_chunks += 1
            n_docs += 1

    manifest = {
        "component": "eval.indexing.rechunk_corpus_jsonl",
        "ts_unix": int(time.time()),
        "duration_sec": int(time.monotonic() - t0),
        "in_jsonl": str(args.in_jsonl),
        "out_jsonl": str(args.out_jsonl),
        "chunker": chunker.name,
        "n_docs": n_docs,
        "n_chunks": n_chunks,
        "chunker_params": {
            "fixed": {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap},
            "parent_child": {
                "parent_size": args.parent_size,
                "parent_overlap": args.parent_overlap,
                "child_size": args.child_size,
                "child_overlap": args.child_overlap,
            },
            "semantic": {
                "embedder_model": args.embedder_model,
                "embedder_device": args.embedder_device,
                "similarity_threshold": args.similarity_threshold,
                "breakpoint_percentile": args.breakpoint_percentile,
                "max_chunk_tokens": args.max_chunk_tokens,
                "min_sentences_per_chunk": args.min_sentences_per_chunk,
            },
        }.get(chunker.name, {}),
    }
    manifest_path = args.out_jsonl.parent / "rechunk_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    print(f"OK: wrote {n_chunks} chunks from {n_docs} docs to {args.out_jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
