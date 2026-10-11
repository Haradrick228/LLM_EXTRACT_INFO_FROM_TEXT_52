"""
Создание / проверка Chroma-коллекций фазы B на одной фикстуре ``chunks_sample.jsonl``.

Примеры:
  python -m eval.phase_b.index_gold_fixture_bases --list
  python -m eval.phase_b.index_gold_fixture_bases --all-st
    # все 6 баз из embedding_bases.yaml (SentenceTransformers / HF)
  python -m eval.phase_b.index_gold_fixture_bases --base-id gold_bge_m3 --rebuild

Артефакты: ``DATA/runs/phase_b_index/<timestamp>/run.log``, ``manifest.json``, ``bases/<base_id>.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.chroma_utils import chroma_http_client  # noqa: E402
from eval.reporting.log_eval_run_config_banner import write_eval_banner  # noqa: E402
from eval.indexing.chroma_embedding_function_factory import build_chroma_embedding_function  # noqa: E402
from eval.datasets.load_gold_fixture_chunks import fixture_path, load_fixture_batches  # noqa: E402
from eval.experiments.phase_b_collections_registry import expand_str, list_bases, load_registry  # noqa: E402

_LOG = logging.getLogger("phase_b.index")


def _run_dir() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return _ROOT / "DATA" / "runs" / "phase_b_index" / ts


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def _merged_base(data: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    m: Dict[str, Any] = dict(data.get("defaults") or {})
    m.update(base)
    return m


def _expand_base_strings(base: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in base.items():
        if isinstance(v, str):
            out[k] = expand_str(v)
        elif isinstance(v, dict):
            out[k] = _expand_base_strings(v)  # type: ignore[assignment]
        elif isinstance(v, list):
            out[k] = [
                expand_str(x) if isinstance(x, str) else x for x in v
            ]
        else:
            out[k] = v
    return out


def _index_one(
    *,
    client,
    merged: Dict[str, Any],
    ids: List[str],
    docs: List[str],
    metas: List[Dict[str, Any]],
    rebuild: bool,
    force_default_fixture: bool,
    log,
) -> Dict[str, Any]:
    base_id = str(merged["base_id"])
    collection = str(merged["collection"])
    protected = bool(merged.get("existing_collection"))
    backend = str(merged.get("backend", ""))
    n = len(ids)

    ef = build_chroma_embedding_function(merged)

    cnt_existing = -1
    try:
        col_ex = client.get_collection(collection, embedding_function=ef)
        cnt_existing = col_ex.count()
    except Exception as e:
        log(f"[{base_id}] get_collection {collection!r}: {e}")

    if not rebuild and cnt_existing == n:
        log(f"[{base_id}] OK: {collection!r} уже на месте, count={cnt_existing}, пропуск.")
        return {"status": "skipped_existing_ok", "collection": collection, "count": cnt_existing}

    if protected and cnt_existing >= 0 and cnt_existing != n and not rebuild:
        log(
            f"[{base_id}] WARN: {collection!r} count={cnt_existing}, ожидалось {n}. "
            "См. scripts/sync_gold_v2_fixture_collection.py или --rebuild --force-include-default-fixture."
        )
        return {"status": "mismatch", "collection": collection, "count": cnt_existing, "expected": n}

    if protected and rebuild and not force_default_fixture:
        raise SystemExit(
            f"Отказ: пересборка защищённой коллекции {collection!r} требует --force-include-default-fixture"
        )

    if cnt_existing >= 0:
        try:
            client.delete_collection(collection)
            log(f"[{base_id}] удалена коллекция {collection!r}")
        except Exception as e:
            log(f"[{base_id}] delete: {e}")

    col = client.create_collection(
        name=collection,
        embedding_function=ef,
        metadata={
            "phase_b_base_id": base_id,
            "fixture": merged.get("fixture_rel", ""),
            "chunking_label": str(merged.get("chunking_label", "")),
            "backend": backend,
            "model_name": str(merged.get("model_name", "")),
        },
    )
    col.add(ids=ids, documents=docs, metadatas=metas)
    cnt = col.count()
    log(f"[{base_id}] создана/заполнена {collection!r}, count={cnt}")
    return {"status": "indexed", "collection": collection, "count": cnt}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Index phase B gold fixture collections (multi-embedder).")
    parser.add_argument("--list", action="store_true", help="Показать base_id и коллекции из YAML.")
    parser.add_argument("--base-id", default="", help="Одна база по base_id.")
    parser.add_argument("--all-st", action="store_true", help="Все базы с backend=sentence_transformers (все 6 в YAML).")
    parser.add_argument("--rebuild", action="store_true", help="Удалить коллекцию перед созданием (кроме защиты default).")
    parser.add_argument(
        "--force-include-default-fixture",
        action="store_true",
        help="Разрешить rebuild для gold_v2_fixture (опасно).",
    )
    parser.add_argument("--registry", type=Path, default=None, help="Путь к embedding_bases.yaml.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_registry(args.registry)

    bases = list_bases(data)
    if args.list:
        for b in bases:
            print(f"{b.get('order')}\t{b.get('base_id')}\t{b.get('collection')}\t{b.get('backend')}\t{b.get('model_name', '')}")
        return 0

    selected: List[Dict[str, Any]] = []
    if args.base_id.strip():
        bid = args.base_id.strip()
        for b in bases:
            if str(b.get("base_id")) == bid:
                selected.append(b)
                break
        if not selected:
            print("Unknown base_id:", bid, file=sys.stderr)
            return 2
    elif args.all_st:
        selected = [b for b in bases if str(b.get("backend", "")).lower() == "sentence_transformers"]
    else:
        parser.error("Укажите --list | --base-id | --all-st")

    run_dir = _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    bases_dir = run_dir / "bases"
    bases_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    log_f = log_path.open("w", encoding="utf-8")
    tee = _Tee(sys.stdout, log_f)

    def log(msg: str) -> None:
        tee.write(msg + "\n")

    host = os.getenv("RAG_EVAL_CHROMA_HOST", os.getenv("CHROMA_HOST", "127.0.0.1"))
    port = int(os.getenv("RAG_EVAL_CHROMA_PORT", os.getenv("CHROMA_PORT", "18000")))
    os.environ.setdefault("CHROMA_HOST", host)
    os.environ.setdefault("CHROMA_PORT", str(port))

    fix_path = _ROOT / str(data.get("defaults", {}).get("fixture_rel", "eval/fixtures/chunks_sample.jsonl"))
    if not fix_path.is_file():
        print("Fixture not found:", fix_path, file=sys.stderr)
        return 2
    fixture_sha = _sha256_file(fix_path)
    ids, docs, metas = load_fixture_batches(fix_path)

    write_eval_banner(
        "phase_b.index_gold_fixture_bases",
        {
            "run_dir": str(run_dir.relative_to(_ROOT)),
            "fixture": str(fix_path.relative_to(_ROOT)),
            "fixture_sha256": fixture_sha,
            "rows": len(ids),
            "fixture_clean_text": True,
            "chroma_host": host,
            "chroma_port": port,
            "rebuild": args.rebuild,
            "selected_base_ids": ", ".join(str(b.get("base_id")) for b in selected),
        },
    )

    log(f"[phase_b_index] run_dir={run_dir}")
    log(f"[phase_b_index] fixture={fix_path} sha256={fixture_sha} rows={len(ids)}")

    client = chroma_http_client(host=host, port=port)
    summary: Dict[str, Any] = {
        "bases": {},
        "fixture_sha256": fixture_sha,
        "fixture_rows": len(ids),
        "fixture_clean_text": True,
    }

    for raw in selected:
        merged = _expand_base_strings(_merged_base(data, raw))
        bid = str(merged["base_id"])
        log(f"\n=== base_id={bid} collection={merged.get('collection')} backend={merged.get('backend')} ===")
        write_eval_banner(
            f"phase_b.index[{bid}]",
            {
                "base_id": bid,
                "collection": merged.get("collection", ""),
                "backend": merged.get("backend", ""),
                "model_name": merged.get("model_name", ""),
                "chunking_label": merged.get("chunking_label", ""),
                "st_device": os.getenv("ST_EMBED_DEVICE", ""),
                "st_batch_size": os.getenv("ST_EMBED_BATCH_SIZE", ""),
            },
        )
        try:
            info = _index_one(
                client=client,
                merged=merged,
                ids=ids,
                docs=docs,
                metas=metas,
                rebuild=args.rebuild,
                force_default_fixture=args.force_include_default_fixture,
                log=log,
            )
        except Exception as e:
            log(f"[{bid}] ERROR: {e}")
            info = {"status": "error", "error": str(e)}
        info["retrieval_env_hint"] = {
            "CHROMA_EMBED_MODEL": str(merged.get("model_name", "")),
            "CHROMA_COLLECTION": str(merged.get("collection", "")),
        }
        summary["bases"][bid] = info
        (bases_dir / f"{bid}.json").write_text(
            json.dumps({"base": merged, "result": info}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (run_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\n[phase_b_index] manifest -> {run_dir / 'manifest.json'}")
    log_f.close()
    return 0 if all(
        summary["bases"].get(b, {}).get("status") in ("indexed", "skipped_existing_ok")
        for b in summary["bases"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
