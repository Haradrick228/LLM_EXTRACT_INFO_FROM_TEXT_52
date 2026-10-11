"""
Индексация полного корпуса: один JSONL (все чанки) → несколько Chroma-коллекций с разными эмбеддерами.

Типичный поток::

    python -m eval.indexing.export_chroma_collection_to_jsonl \
        --collection default_clean --run-id corpus_export_001
    python -m eval.indexing.index_corpus_from_jsonl \
        --jsonl DATA/runs/corpus_export_001/chunks_from_chroma.jsonl --all-st --rebuild

Индексация ``full_giga_embeddings_instruct`` по умолчанию **запрещена**; явно: ``PHASE_B_ALLOW_FULL_GIGA=1``.

Артефакты: ``DATA/runs/phase_b_index_full/<timestamp>/run.log``, ``manifest.json``, ``bases/<base_id>.json``.
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
from eval.indexing.jsonl_corpus_stream import count_nonempty_lines, iter_jsonl_batches  # noqa: E402
from eval.experiments.phase_b_collections_registry import expand_str, list_bases, load_registry  # noqa: E402

_DEFAULT_REGISTRY = _ROOT / "eval" / "config" / "embedding_bases_full_corpus.yaml"
if not _DEFAULT_REGISTRY.exists():
    _DEFAULT_REGISTRY = _ROOT / "eval" / "phase_b" / "embedding_bases_full_corpus.yaml"


def _run_dir() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return _ROOT / "DATA" / "runs" / "phase_b_index_full" / ts


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
            out[k] = [expand_str(x) if isinstance(x, str) else x for x in v]
        else:
            out[k] = v
    return out


def _add_batch_size() -> int:
    return max(1, int(os.getenv("PHASE_B_CHROMA_ADD_BATCH", "256") or 256))


def _apply_full_giga_index_policy(
    selected: List[Dict[str, Any]],
    *,
    explicit_base_id: str,
    all_st: bool,
) -> List[Dict[str, Any]]:
    """
    При ``--all-st`` по умолчанию пропускаем Giga (тяжёлая); явный ``--base-id full_giga_*`` всегда разрешён.

    Чтобы включить Giga в ``--all-st``: ``PHASE_B_ALLOW_FULL_GIGA=1``.
    """
    allow = os.getenv("PHASE_B_ALLOW_FULL_GIGA", "").strip() == "1"
    explicit_giga = explicit_base_id.strip() == "full_giga_embeddings_instruct"
    if explicit_giga or allow:
        return selected
    if not all_st:
        return selected
    filtered = [b for b in selected if str(b.get("base_id")) != "full_giga_embeddings_instruct"]
    if len(filtered) < len(selected):
        print(
            "[phase_b.index_jsonl_bases] Пропуск full_giga_embeddings_instruct в --all-st "
            "(нет PHASE_B_ALLOW_FULL_GIGA=1). Остальные базы выполняются.",
            file=sys.stderr,
        )
    return filtered


_QWEN3_BASE_IDS = ("full_qwen3_embedding_4b", "full_qwen3_embedding_8b")


def _apply_full_qwen3_index_policy(
    selected: List[Dict[str, Any]],
    *,
    explicit_base_id: str,
    all_st: bool,
) -> List[Dict[str, Any]]:
    """Аналогично Giga, но для семейства Qwen3 (PHASE_B_ALLOW_FULL_QWEN3=1).

    Qwen3-Embedding-8B требует ~16-18 GB VRAM (fp16), 4B — ~10 GB. По умолчанию
    в ``--all-st`` не берём; явный ``--base-id full_qwen3_*`` всегда разрешён.
    """
    allow = os.getenv("PHASE_B_ALLOW_FULL_QWEN3", "").strip() == "1"
    explicit_qwen = explicit_base_id.strip() in _QWEN3_BASE_IDS
    if explicit_qwen or allow:
        return selected
    if not all_st:
        return selected
    filtered = [b for b in selected if str(b.get("base_id")) not in _QWEN3_BASE_IDS]
    if len(filtered) < len(selected):
        print(
            "[phase_b.index_jsonl_bases] Пропуск Qwen3-Embedding-* в --all-st "
            "(нет PHASE_B_ALLOW_FULL_QWEN3=1). Остальные базы выполняются.",
            file=sys.stderr,
        )
    return filtered


def _index_streaming(
    *,
    client,
    merged: Dict[str, Any],
    jsonl_path: Path,
    rebuild: bool,
    resume: bool,
    apply_clean_text: bool,
    max_rows: Optional[int],
    add_batch: int,
    log,
) -> Dict[str, Any]:
    base_id = str(merged["base_id"])
    collection = str(merged["collection"])
    protected = bool(merged.get("existing_collection"))
    backend = str(merged.get("backend", ""))

    total_lines = count_nonempty_lines(jsonl_path)
    n_target = total_lines if max_rows is None else min(max_rows, total_lines)

    ef = build_chroma_embedding_function(merged)

    cnt_existing = -1
    try:
        col_ex = client.get_collection(collection, embedding_function=ef)
        cnt_existing = col_ex.count()
    except Exception as e:
        log(f"[{base_id}] get_collection {collection!r}: {e}")

    if not rebuild and cnt_existing == n_target:
        log(f"[{base_id}] OK: {collection!r} уже на месте, count={cnt_existing}, пропуск.")
        return {"status": "skipped_existing_ok", "collection": collection, "count": cnt_existing}

    if protected and cnt_existing >= 0 and cnt_existing != n_target and not rebuild:
        log(
            f"[{base_id}] WARN: {collection!r} count={cnt_existing}, ожидалось {n_target}. "
            "Используйте --rebuild."
        )
        return {"status": "mismatch", "collection": collection, "count": cnt_existing, "expected": n_target}

    if protected and rebuild:
        raise SystemExit(
            f"Отказ: коллекция {collection!r} помечена existing_collection; для полного корпуса не используйте этот флаг."
        )

    if cnt_existing >= 0:
        try:
            client.delete_collection(collection)
            log(f"[{base_id}] удалена коллекция {collection!r}")
        except Exception as e:
            log(f"[{base_id}] delete: {e}")

    try:
        jsonl_rel = str(jsonl_path.relative_to(_ROOT))
    except ValueError:
        jsonl_rel = str(jsonl_path)

    col = client.create_collection(
        name=collection,
        embedding_function=ef,
        metadata={
            "phase_b_base_id": base_id,
            "source_jsonl": jsonl_rel,
            "chunking_label": str(merged.get("chunking_label", "")),
            "backend": backend,
            "model_name": str(merged.get("model_name", "")),
            "expected_rows": str(n_target),
        },
    )

    done = 0
    log_interval = max(add_batch, min(10_000, max(n_target // 25, 1)))
    last_logged = 0
    for ids, docs, metas in iter_jsonl_batches(
        jsonl_path,
        add_batch,
        apply_clean_text=apply_clean_text,
        max_rows=max_rows,
    ):
        col.add(ids=ids, documents=docs, metadatas=metas)
        done += len(ids)
        if done == n_target or done - last_logged >= log_interval:
            log(f"[{base_id}] progress add … rows={done}/{n_target}")
            last_logged = done

    cnt = col.count()
    log(f"[{base_id}] создана/заполнена {collection!r}, count={cnt} (ожидали {n_target})")
    if cnt != n_target:
        return {"status": "count_mismatch", "collection": collection, "count": cnt, "expected": n_target}
    return {"status": "indexed", "collection": collection, "count": cnt}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Index full-corpus JSONL into multiple Chroma collections (phase B).")
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Путь к JSONL: id, document, metadata (не нужен при --list).",
    )
    parser.add_argument("--list", action="store_true", help="Показать base_id и коллекции из YAML.")
    parser.add_argument("--base-id", default="", help="Одна база по base_id.")
    parser.add_argument("--all-st", action="store_true", help="Все базы с backend=sentence_transformers.")
    parser.add_argument("--rebuild", action="store_true", help="Удалить коллекцию перед заполнением.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=_DEFAULT_REGISTRY,
        help="YAML реестра (по умолчанию embedding_bases_full_corpus.yaml).",
    )
    parser.add_argument("--max-rows", type=int, default=0, help="Ограничить число строк (0 = без лимита).")
    parser.add_argument(
        "--no-clean-text",
        action="store_true",
        help="Не применять eval.text_pipeline.clean_text к document.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_registry(args.registry)

    bases = list_bases(data)
    if args.list:
        for b in bases:
            print(f"{b.get('order')}\t{b.get('base_id')}\t{b.get('collection')}\t{b.get('backend')}\t{b.get('model_name', '')}")
        return 0

    if args.jsonl is None:
        parser.error("Укажите --jsonl (или используйте --list)")

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

    selected = _apply_full_giga_index_policy(
        selected, explicit_base_id=args.base_id, all_st=bool(args.all_st),
    )
    selected = _apply_full_qwen3_index_policy(
        selected, explicit_base_id=args.base_id, all_st=bool(args.all_st),
    )
    if not selected:
        print(
            "[phase_b.index_jsonl_bases] Нет баз для индексации после политик Giga/Qwen3.",
            file=sys.stderr,
        )
        return 0

    jsonl_path = args.jsonl.resolve()
    if not jsonl_path.is_file():
        print("JSONL not found:", jsonl_path, file=sys.stderr)
        return 2

    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None
    apply_clean = not args.no_clean_text
    add_batch = _add_batch_size()

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

    file_sha = _sha256_file(jsonl_path)
    n_lines = count_nonempty_lines(jsonl_path)
    n_target = n_lines if max_rows is None else min(max_rows, n_lines)

    write_eval_banner(
        "phase_b.index_jsonl_bases",
        {
            "run_dir": str(run_dir.relative_to(_ROOT)),
            "jsonl": str(jsonl_path.relative_to(_ROOT)) if jsonl_path.is_relative_to(_ROOT) else str(jsonl_path),
            "jsonl_sha256": file_sha,
            "jsonl_nonempty_lines": n_lines,
            "index_target_rows": n_target,
            "apply_clean_text": apply_clean,
            "add_batch": add_batch,
            "chroma_host": host,
            "chroma_port": port,
            "rebuild": args.rebuild,
            "registry": str(args.registry),
            "selected_base_ids": ", ".join(str(b.get("base_id")) for b in selected),
        },
    )

    log(f"[phase_b_index_full] run_dir={run_dir}")
    log(f"[phase_b_index_full] jsonl={jsonl_path} sha256={file_sha} lines={n_lines} target_rows={n_target}")

    client = chroma_http_client(host=host, port=port)
    summary: Dict[str, Any] = {
        "bases": {},
        "source_jsonl_sha256": file_sha,
        "source_jsonl_lines": n_lines,
        "index_target_rows": n_target,
        "apply_clean_text": apply_clean,
        "add_batch": add_batch,
    }

    for raw in selected:
        merged = _expand_base_strings(_merged_base(data, raw))
        bid = str(merged["base_id"])
        log(f"\n=== base_id={bid} collection={merged.get('collection')} backend={merged.get('backend')} ===")
        write_eval_banner(
            f"phase_b.index_full[{bid}]",
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
            info = _index_streaming(
                client=client,
                merged=merged,
                jsonl_path=jsonl_path,
                rebuild=args.rebuild,
                resume=False,
                apply_clean_text=apply_clean,
                max_rows=max_rows,
                add_batch=add_batch,
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
    log(f"\n[phase_b_index_full] manifest -> {run_dir / 'manifest.json'}")
    log_f.close()
    return 0 if all(
        summary["bases"].get(b, {}).get("status") in ("indexed", "skipped_existing_ok")
        for b in summary["bases"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
