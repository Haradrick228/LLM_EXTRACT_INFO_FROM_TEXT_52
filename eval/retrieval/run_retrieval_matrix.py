"""
B3 (после полной индексации corpus_phase_b_*): матрица по dense-коллекциям из ``embedding_bases_full_corpus.yaml``.

По умолчанию: **реранк** — профили ``rag_profiles.yaml`` (matrix: true) × каждая коллекция
(dense-retrieval в Chroma, затем CrossEncoder).

``--retrieval-only``: только **dense-retrieval** (как вернул Chroma), без реранка — проверка пяти эмбеддеров до матрицы реранкеров.

  python -m eval.phase_b.run_full_corpus_rerank_matrix --expected-rows 61543
  python -m eval.phase_b.run_full_corpus_rerank_matrix --expected-rows 61543 --retrieval-max-rows 3 --retrieval-trace --skip-answer

Артефакты: ``eval/runs/<ts>_b3_full_corpus_rerank|retrieval/``, ``summary.json``, ``matrix_manifest.json``,
``<scenario_id>/retrieval_*.json``,
при ``--retrieval-trace``: ``<scenario_id>/retrieval_trace_<profile>.jsonl`` (JSONL: run_header, query_trace × N, run_footer).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.rag_profile_env import apply_profile, list_profile_names  # noqa: E402
from eval.chroma_utils import chroma_http_client  # noqa: E402
from eval.reporting.log_eval_run_config_banner import write_eval_banner  # noqa: E402
from eval.experiments.phase_b_collections_registry import list_bases, load_registry  # noqa: E402


def _merged_base(data: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    m: Dict[str, Any] = dict(data.get("defaults") or {})
    m.update(base)
    return m


def _collection_fallback_for_merged(merged: Dict[str, Any]) -> Optional[str]:
    fb = merged.get("collection_fallback")
    if isinstance(fb, str) and fb.strip():
        return fb.strip()
    return None


def _expected_rows_for_merged(merged: Dict[str, Any], global_expected: int) -> Optional[int]:
    """None — не проверять count()."""
    if "expected_collection_rows" not in merged:
        return global_expected if global_expected > 0 else None
    raw = merged.get("expected_collection_rows")
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() == "skip":
        return None
    return int(raw)


def _scenario_rows(registry: Path) -> List[Tuple[str, str, str, Optional[str]]]:
    """(base_id, collection as в YAML, embed_model, collection_fallback|None)."""
    data = load_registry(registry)
    out: List[Tuple[str, str, str, Optional[str]]] = []
    for b in list_bases(data):
        merged = _merged_base(data, b)
        bid = str(merged["base_id"])
        col = str(merged["collection"])
        model = str(merged.get("model_name", "")).strip()
        if not model:
            raise ValueError(f"base_id={bid} missing model_name")
        out.append((bid, col, model, _collection_fallback_for_merged(merged)))
    return out


def _merged_for_base_id(data: Dict[str, Any], base_id: str) -> Dict[str, Any]:
    for b in list_bases(data):
        m = _merged_base(data, b)
        if str(m["base_id"]) == base_id:
            return m
    raise ValueError(f"base_id {base_id!r} not in registry")


def _metric_keys_from_payload(payload: Dict[str, Any]) -> List[str]:
    """Собрать ключи метрик из одного retrieval JSON (без _meta), стабильный порядок."""
    keys: List[str] = []
    for k in sorted(payload.keys()):
        if k == "_meta":
            continue
        v = payload[k]
        if not isinstance(v, (int, float)):
            continue
        keys.append(k)
    return keys


def _discover_matrix_metric_keys(run_root: Path, summary: Dict[str, Any], jobs: List[str]) -> List[str]:
    seen: Dict[str, None] = {}
    scenarios: List[str] = list(summary.get("scenarios") or [])
    for sid in scenarios:
        for job_id in jobs:
            pth = run_root / sid / f"retrieval_{job_id}.json"
            if not pth.is_file():
                continue
            try:
                raw = json.loads(pth.read_text(encoding="utf-8"))
            except Exception:
                continue
            for k in _metric_keys_from_payload(raw):
                seen.setdefault(k, None)
    # Primary metrics first (no mode prefix), then lexicographic for token_f1:: / semantic::
    primary = [k for k in seen if "::" not in k]
    secondary = [k for k in seen if "::" in k]
    primary_sorted = sorted(primary, key=lambda x: (0 if x.startswith("recall@") else 1, x))
    secondary_sorted = sorted(secondary)
    return primary_sorted + secondary_sorted


def _write_metrics_matrix_artifacts(
    run_root: Path,
    summary: Dict[str, Any],
    *,
    profile_names: List[str],
    retrieval_only: bool,
) -> None:
    """Вшить метрики из retrieval_*.json в summary и записать markdown-таблицы."""
    scenarios: List[str] = list(summary.get("scenarios") or [])
    jobs: List[str] = (
        ["dense_only"]
        if retrieval_only
        else list(profile_names)
    )

    for sid in scenarios:
        bucket = summary.get("results", {}).get(sid) or {}
        for job_id in jobs:
            cell = bucket.get(job_id) or {}
            if cell.get("retrieval_returncode", 1) != 0:
                continue
            pth = run_root / sid / f"retrieval_{job_id}.json"
            if not pth.is_file():
                continue
            raw = json.loads(pth.read_text(encoding="utf-8"))
            meta = raw.pop("_meta", {})
            cell["metrics"] = dict(raw)
            cell["metrics_meta"] = meta

    metric_keys = _discover_matrix_metric_keys(run_root, summary, jobs)
    if not metric_keys:
        metric_keys = ["recall@5", "recall@10", "precision@5", "precision@10", "mrr", "ndcg@3"]

    lines: List[str] = [
        "# B3 retrieval metrics matrix",
        "",
        f"Run directory: `{run_root}`",
        "",
        "Метрики из `run_retrieval_eval` (после реранка, если профиль его задаёт). "
        "При `--metric-modes substring,token_f1,semantic` появляются ключи вида `token_f1::recall@10`.",
        "",
    ]
    for mk in metric_keys:
        lines.append(f"## {mk}")
        lines.append("")
        header = "| scenario | " + " | ".join(jobs) + " |"
        sep = "|" + "|".join(["---"] * (len(jobs) + 1)) + "|"
        lines.extend([header, sep])
        for sid in scenarios:
            row_cells: List[str] = []
            for job_id in jobs:
                cell = (summary.get("results") or {}).get(sid, {}).get(job_id) or {}
                m = cell.get("metrics") or {}
                v = m.get(mk)
                row_cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "")
            lines.append("| " + sid + " | " + " | ".join(row_cells) + " |")
        lines.append("")

    (run_root / "metrics_matrix.md").write_text("\n".join(lines), encoding="utf-8")


def _resolve_chroma_collection(names: set[str], primary: str, fallback: Optional[str]) -> str:
    if primary in names:
        return primary
    if fallback and fallback in names:
        print(
            f"[matrix] chroma_collection={primary!r} missing; using collection_fallback={fallback!r}",
            file=sys.stderr,
        )
        return fallback
    sample = ", ".join(sorted(names)[:40])
    raise ValueError(
        f"Chroma has neither collection={primary!r} nor fallback={fallback!r}. "
        f"Collections (sample): {sample}"
    )


def _verify_collection_count_merged(
    merged: Dict[str, Any], resolved_collection: str, host: str, port: str
) -> int:
    """count() коллекции без загрузки embedding function.

    Для count() embedding function не требуется; избегаем лишних HF-запросов
    при валидации матрицы перед запуском ячеек.
    """
    client = chroma_http_client(host=host, port=int(port))
    col = client.get_collection(resolved_collection)
    return int(col.count())


def main() -> int:
    p = argparse.ArgumentParser(description="B3: rerank matrix over full-corpus dense collections + gold eval file.")
    p.add_argument(
        "--dense-registry",
        type=Path,
        default=_ROOT / "eval" / "config" / "embedding_bases_full_corpus.yaml",
        help="YAML с коллекциями corpus_phase_b_* (как после index_jsonl_bases).",
    )
    p.add_argument(
        "--profiles",
        default="matrix",
        help="'matrix' или список имён профилей через запятую (rag_profiles.yaml).",
    )
    p.add_argument(
        "--eval-file",
        default=str(_ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl"),
    )
    p.add_argument("--fetch-k", type=int, default=int(os.getenv("RETRIEVER_FETCH_K", "40") or 40))
    p.add_argument("--k", default="5,10")
    p.add_argument("--skip-answer", action="store_true")
    p.add_argument(
        "--retrieval-max-rows",
        type=int,
        default=int(os.getenv("RETRIEVAL_EVAL_MAX_ROWS", "0") or 0),
        help="0 = все вопросы в eval-файле (gold v2 = 101).",
    )
    p.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Только dense-retrieval (без реранка): один прогон run_retrieval_eval на сценарий, "
        "без профилей rag_profiles. Порядок документов — как у Chroma query.",
    )
    p.add_argument(
        "--expected-rows",
        type=int,
        default=0,
        help="Глобальный ожидаемый count() для баз без поля expected_collection_rows в YAML. "
        "Пер-базовый expected задаётся в YAML (expected_collection_rows).",
    )
    p.add_argument(
        "--retrieval-trace",
        action="store_true",
        help="Писать JSONL-трейс по каждому вопросу: retrieval_trace_<job>.jsonl в каталоге сценария.",
    )
    p.add_argument(
        "--trace-doc-chars",
        type=int,
        default=int(os.getenv("RETRIEVAL_TRACE_DOC_CHARS", "2500") or 2500),
        help="Символов текста чанка в трейсе; 0 = без обрезки.",
    )
    p.add_argument(
        "--skip-validate-gold-v2",
        action="store_true",
        help="Не запускать python -m eval.validate_gold_v2 перед матрицей.",
    )
    p.add_argument(
        "--base-ids",
        default="",
        help="Через запятую: только эти base_id из registry (например шесть рабочих без Giga/Qwen).",
    )
    p.add_argument(
        "--metric-modes",
        default=(os.getenv("RETRIEVAL_METRIC_MODES", "substring,token_f1,semantic") or "substring,token_f1,semantic").strip(),
        help="Передать в run_retrieval_eval --metric-modes (primary = первый режим, для A–D в лестнице).",
    )
    p.add_argument(
        "--semantic-threshold",
        type=float,
        default=float(os.getenv("RETRIEVAL_SEMANTIC_THRESHOLD", "0.62") or 0.62),
        help="Порог cosine для semantic hit (если semantic в --metric-modes).",
    )
    p.add_argument(
        "--semantic-model",
        default=os.getenv(
            "RETRIEVAL_SEMANTIC_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ).strip(),
        help="Модель для semantic hit.",
    )
    args = p.parse_args()

    if args.retrieval_only:
        profile_names: List[str] = []
    elif args.profiles.strip() == "matrix":
        profile_names = list_profile_names(matrix_only=True)
    else:
        profile_names = [x.strip() for x in args.profiles.split(",") if x.strip()]

    if not args.retrieval_only and not profile_names:
        print("No profiles to run (use --profiles or matrix).", file=sys.stderr)
        return 2

    scenarios = _scenario_rows(args.dense_registry.resolve())
    if args.base_ids.strip():
        wanted = {x.strip() for x in args.base_ids.split(",") if x.strip()}
        scenarios = [s for s in scenarios if s[0] in wanted]
    if not scenarios:
        print("No scenarios in registry", args.dense_registry, file=sys.stderr)
        return 2

    reg_data = load_registry(args.dense_registry.resolve())

    baseline = dict(os.environ)
    eval_chroma_host = baseline.get("RAG_EVAL_CHROMA_HOST", "127.0.0.1")
    eval_chroma_port = baseline.get("RAG_EVAL_CHROMA_PORT", "18000")

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_slug = "b3_full_corpus_retrieval" if args.retrieval_only else "b3_full_corpus_rerank"
    run_root = _ROOT / "eval" / "runs" / f"{ts}_{run_slug}"
    run_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_validate_gold_v2:
        vcmd = [sys.executable, "-m", "eval.validate_gold_v2"]
        vproc = subprocess.run(vcmd, cwd=str(_ROOT), capture_output=True, text=True)
        (run_root / "validate_gold_v2.stdout.txt").write_text(vproc.stdout or "", encoding="utf-8")
        (run_root / "validate_gold_v2.stderr.txt").write_text(vproc.stderr or "", encoding="utf-8")
        if vproc.returncode != 0:
            print(
                "ERR: eval.validate_gold_v2 failed (fixture vs gold). See",
                run_root / "validate_gold_v2.stderr.txt",
                file=sys.stderr,
            )
            return 2

    chroma_names = {c.name for c in chroma_http_client(host=eval_chroma_host, port=int(eval_chroma_port)).list_collections()}

    scenario_manifest: List[Dict[str, Any]] = []
    for sid, col_yaml, embed_model, fb in scenarios:
        merged = _merged_for_base_id(reg_data, sid)
        resolved = _resolve_chroma_collection(chroma_names, col_yaml, fb)
        exp = _expected_rows_for_merged(merged, args.expected_rows)
        count_verified: Optional[int] = None
        if exp is not None:
            n = _verify_collection_count_merged(merged, resolved, eval_chroma_host, eval_chroma_port)
            count_verified = n
            if n != exp:
                print(
                    f"ERR scenario={sid} collection_yaml={col_yaml!r} collection_resolved={resolved!r} "
                    f"count={n} expected={exp}",
                    file=sys.stderr,
                )
                return 2
        scenario_manifest.append(
            {
                "base_id": sid,
                "chroma_collection_yaml": col_yaml,
                "chroma_collection_resolved": resolved,
                "collection_fallback": fb,
                "embed_model": embed_model,
                "trust_remote_code": bool((merged.get("st_model_kwargs") or {}).get("trust_remote_code")),
                "expected_collection_rows": exp,
                "chroma_count_verified": count_verified,
            }
        )

    matrix_manifest: Dict[str, Any] = {
        "component": "phase_b.run_full_corpus_rerank_matrix",
        "ts": ts,
        "eval_file": args.eval_file,
        "b3_base_ids": [s[0] for s in scenarios],
        "metric_modes": args.metric_modes,
        "semantic_threshold": float(args.semantic_threshold),
        "semantic_model": args.semantic_model,
        "eval_scenario_type": os.getenv("EVAL_SCENARIO_TYPE", "full_corpus"),
        "run_retrieval_eval_contract": {
            "relevance_fields": ["question", "references", "reference_quote (если references пуст)"],
            "note": "Метрики: для каждого режима из --metric-modes пишутся ключи вида mode::recall@k; primary (первый режим) также дублируется без префикса для обратной совместимости. answer_ref в retrieval не участвует.",
        },
        "evaluation_reproducibility": {
            "RETRIEVAL_EVAL_SHUFFLE": "0",
            "note": "В каждом subprocess задаётся RETRIEVAL_EVAL_SHUFFLE=0: порядок вопросов = порядок строк в JSONL. "
            "Иначе load_eval использует random.shuffle и средние метрики «плавают» между сценариями и прогонами.",
        },
        "metrics_tables": "metrics_matrix.md (после завершения всех ячеек)",
        "chroma_eval_host": eval_chroma_host,
        "chroma_eval_port": int(eval_chroma_port),
        "scenarios": scenario_manifest,
        "profiles_matrix": [] if args.retrieval_only else profile_names,
    }
    (run_root / "matrix_manifest.json").write_text(
        json.dumps(matrix_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for line in json.dumps(scenario_manifest, ensure_ascii=False, indent=2).splitlines():
        print("[matrix_manifest]", line, flush=True)

    summary: Dict[str, Any] = {
        "run_dir": str(run_root.relative_to(_ROOT)),
        "dense_registry": str(args.dense_registry),
        "scenarios": [s[0] for s in scenarios],
        "retrieval_only": args.retrieval_only,
        "profiles": profile_names,
        "eval_file": args.eval_file,
        "fetch_k": args.fetch_k,
        "k": args.k,
        "metric_modes": args.metric_modes,
        "semantic_threshold": float(args.semantic_threshold),
        "semantic_model": args.semantic_model,
        "eval_scenario_type": os.getenv("EVAL_SCENARIO_TYPE", "full_corpus"),
        "replay_hint": "python -m eval.replay_experiment run phase_b3_full_corpus_rerank_matrix",
        "expected_rows_cli": args.expected_rows,
        "retrieval_trace": args.retrieval_trace,
        "trace_doc_chars": args.trace_doc_chars,
        "matrix_manifest": "matrix_manifest.json",
        "metrics_matrix_md": "metrics_matrix.md",
        "scenario_chroma": {row["base_id"]: row for row in scenario_manifest},
        "results": {},
    }

    prof_banner = "(dense-only, no rerank)" if args.retrieval_only else ", ".join(profile_names)
    write_eval_banner(
        "phase_b.run_full_corpus_rerank_matrix",
        {
            "run_dir": str(run_root.relative_to(_ROOT)),
            "dense_registry": str(args.dense_registry),
            "scenarios": ", ".join(s[0] for s in scenarios),
            "profiles": prof_banner,
            "eval_file": args.eval_file,
            "fetch_k": args.fetch_k,
            "metric_modes": args.metric_modes,
            "semantic_threshold": str(args.semantic_threshold),
            "semantic_model": args.semantic_model,
            "eval_scenario_type": os.getenv("EVAL_SCENARIO_TYPE", "full_corpus"),
            "expected_rows_cli": args.expected_rows,
            "retrieval_only": str(args.retrieval_only),
            "retrieval_trace": str(args.retrieval_trace),
            "matrix_manifest": str(run_root / "matrix_manifest.json"),
        },
    )

    all_ok = True
    for sid, col_yaml, embed_model, _fb in scenarios:
        sm = next(x for x in scenario_manifest if x["base_id"] == sid)
        resolved = str(sm["chroma_collection_resolved"])
        merged = _merged_for_base_id(reg_data, sid)

        scen_dir = run_root / sid
        scen_dir.mkdir(parents=True, exist_ok=True)
        summary["results"][sid] = {}

        scen_baseline = dict(baseline)
        scen_baseline["CHROMA_COLLECTION"] = resolved
        scen_baseline["CHROMA_EMBED_MODEL"] = embed_model
        scen_baseline["RETRIEVAL_EVAL_SHUFFLE"] = "0"
        scen_baseline.setdefault("EVAL_SCENARIO_TYPE", "full_corpus")
        scen_baseline["EVAL_SEED"] = scen_baseline.get("EVAL_SEED", "0")

        if args.retrieval_only:
            run_jobs: List[Tuple[str, str, str]] = [
                ("dense_only", "", "none"),
            ]
        else:
            run_jobs = [(pname, "", "") for pname in profile_names]

        for job_id, _unused_model, _unused_mode in run_jobs:
            os.environ.clear()
            os.environ.update(scen_baseline)
            if args.retrieval_only:
                for k in ("RERANK_MODEL", "RERANK_MODE", "RERANK_DEVICE"):
                    os.environ.pop(k, None)
                rerank_model = ""
                rerank_mode = "none"
                applied: Dict[str, str] = {}
                (scen_dir / "retrieval_dense_only_env.json").write_text(
                    json.dumps({"note": "no rag_profile; rerank off"}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            else:
                pname = job_id
                applied = apply_profile(pname)
                (scen_dir / f"profile_{pname}_env_applied.json").write_text(
                    json.dumps(applied, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                rerank_model = os.getenv("RERANK_MODEL", "")
                rerank_mode = os.getenv("RERANK_MODE", "auto")

            preamble = {
                "component": "phase_b.run_full_corpus_rerank_matrix.subprocess",
                "base_id": sid,
                "chroma_collection_yaml": col_yaml,
                "chroma_collection_resolved": resolved,
                "embed_model": embed_model,
                "rerank_model": rerank_model,
                "rerank_mode": rerank_mode,
                "job_id": job_id,
            }
            print("[matrix_job]", json.dumps(preamble, ensure_ascii=False), flush=True)

            out_json = scen_dir / f"retrieval_{job_id}.json"
            cmd = [
                sys.executable,
                "-m",
                "eval.run_retrieval_eval",
                "--eval-file",
                args.eval_file,
                "--embed-model",
                embed_model,
                "--fetch-k",
                str(args.fetch_k),
                "--k",
                args.k,
                "--metric-modes",
                args.metric_modes,
                "--semantic-threshold",
                str(args.semantic_threshold),
                "--semantic-model",
                args.semantic_model,
                "--rerank-model",
                rerank_model,
                "--rerank-mode",
                rerank_mode,
                "--chroma-host",
                eval_chroma_host,
                "--chroma-port",
                str(eval_chroma_port),
                "--chroma-collection",
                resolved,
                "--out-json",
                str(out_json),
            ]
            st_kw = (merged.get("st_model_kwargs") or {})
            if bool(st_kw.get("trust_remote_code")):
                cmd.append("--embed-trust-remote-code")
            if args.retrieval_max_rows > 0:
                cmd.extend(["--max-rows", str(args.retrieval_max_rows)])
            trace_rel: str | None = None
            if args.retrieval_trace:
                tpath = scen_dir / f"retrieval_trace_{job_id}.jsonl"
                trace_rel = str(tpath.relative_to(_ROOT))
                cmd.extend(
                    [
                        "--retrieval-trace-jsonl",
                        str(tpath),
                        "--trace-doc-chars",
                        str(args.trace_doc_chars),
                        "--trace-profile-label",
                        job_id,
                    ]
                )

            proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
            (scen_dir / f"retrieval_{job_id}.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
            (scen_dir / f"retrieval_{job_id}.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
            row: Dict[str, Any] = {
                "retrieval_returncode": proc.returncode,
                "chroma_collection_resolved": resolved,
                "chroma_collection_yaml": col_yaml,
                "embed_model": embed_model,
                "rerank_model": rerank_model,
                "rerank_mode": rerank_mode,
            }
            if trace_rel:
                row["retrieval_trace_jsonl"] = trace_rel
            summary["results"][sid][job_id] = row
            all_ok = all_ok and proc.returncode == 0

            if not args.skip_answer and proc.returncode == 0 and out_json.is_file():
                ans_dir = scen_dir / f"_answer_{job_id}"
                ans_dir.mkdir(parents=True, exist_ok=True)
                proc_a = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "eval.diagnostics.run_answer_eval_in_run_dir",
                        "--run-dir",
                        str(ans_dir),
                        "--retrieval-json",
                        str(out_json),
                        "--max-rows",
                        str(args.retrieval_max_rows),
                    ],
                    cwd=str(_ROOT),
                    capture_output=True,
                    text=True,
                )
                (scen_dir / f"answer_{job_id}.stdout.txt").write_text(proc_a.stdout or "", encoding="utf-8")
                (scen_dir / f"answer_{job_id}.stderr.txt").write_text(proc_a.stderr or "", encoding="utf-8")
                summary["results"][sid][job_id]["answer_returncode"] = proc_a.returncode
                summary["results"][sid][job_id]["answer_run_dir"] = str(ans_dir.relative_to(_ROOT))
                all_ok = all_ok and proc_a.returncode == 0

    os.environ.clear()
    os.environ.update(baseline)

    _write_metrics_matrix_artifacts(
        run_root,
        summary,
        profile_names=profile_names,
        retrieval_only=bool(args.retrieval_only),
    )

    (run_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done. Artifacts in", run_root, "| metrics_matrix.md | summary.json")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
