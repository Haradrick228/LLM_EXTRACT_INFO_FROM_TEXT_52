"""Пилот классификации ошибок RAG: retrieval (type1) vs generation (type2).

Сопоставляет ``query_trace`` из retrieval JSONL с optional ``answer_eval_results.jsonl``.
См. ``pipeline_trace_v1`` в emit_rag_observability_report (фаза C — запись из RagService).

Usage::

    python -m eval.diagnostics.classify_rag_errors \\
        --trace eval/runs/<id>/trace.jsonl \\
        --gold-jsonl eval/datasets/eval_set_gold_v2.jsonl \\
        --answer-jsonl eval/runs/<id>/_answer_top/.../answer_eval_results.jsonl \\
        --out-json eval/runs/<id>/rag_error_classification.json \\
        --out-per-query-jsonl eval/runs/<id>/rag_error_per_query.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_SCHEMA = "rag_error_classification_v1"


def _load_gold_by_index(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                out[i] = json.loads(line)
    return out


def _question_key(q: str) -> str:
    return " ".join((q or "").split())


def _load_answer_by_index(
    path: Optional[Path], gold: Dict[int, Dict[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    if path is None or not path.is_file():
        return {}
    q_to_idx = {_question_key(str(row.get("question") or "")): i for i, row in gold.items()}
    out: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qk = _question_key(str(row.get("question") or ""))
            idx = q_to_idx.get(qk)
            if idx is not None:
                out[idx] = row
    return out


def _load_ladder_per_row(path: Optional[Path]) -> Dict[int, str]:
    """Optional: {\"per_row\": [{\"eval_row_index\": 0, \"class\": \"A\"}, ...]}."""
    if path is None or not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("per_row")
    if not isinstance(rows, list):
        return {}
    out: Dict[int, str] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        idx = item.get("eval_row_index")
        clazz = item.get("class")
        if idx is not None and clazz:
            out[int(idx)] = str(clazz)
    return out


def _iter_query_traces(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("record") == "query_trace":
                yield o


def _recall_at_k(trace_row: Dict[str, Any], primary: str, k: int) -> Optional[float]:
    mm = trace_row.get("metrics_by_mode") or {}
    block = mm.get(primary) or {}
    key = f"recall@{k}"
    v = block.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _classify_one(
    *,
    eval_row_index: int,
    recall_k: Optional[float],
    ref_sim: Optional[float],
    ladder_class: Optional[str],
    ref_sim_threshold: float,
    has_answer_eval: bool,
) -> str:
    ladder_type1 = ladder_class in ("A", "B")
    retrieval_miss = recall_k is not None and recall_k <= 0.0
    retrieval_hit = recall_k is not None and recall_k > 0.0

    if ladder_type1 or retrieval_miss:
        return "type1_retrieval"

    if retrieval_hit:
        if not has_answer_eval:
            return "type0_ok"
        if ref_sim is None:
            return "unknown"
        if ref_sim < ref_sim_threshold:
            return "type2_generation"
        return "type0_ok"

    return "unknown"


def classify_rag_errors(
    *,
    trace_path: Path,
    gold_path: Path,
    answer_path: Optional[Path] = None,
    ladder_json: Optional[Path] = None,
    primary: str = "substring",
    k: int = 10,
    ref_sim_threshold: float = 0.55,
) -> Dict[str, Any]:
    gold = _load_gold_by_index(gold_path)
    answers = _load_answer_by_index(answer_path, gold)
    ladder_per_row = _load_ladder_per_row(ladder_json)
    has_answer_eval = answer_path is not None and answer_path.is_file()

    per_query: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)

    for trace_row in _iter_query_traces(trace_path):
        idx = int(trace_row.get("eval_row_index", -1))
        recall_k = _recall_at_k(trace_row, primary, k)
        ans = answers.get(idx) or {}
        ref_sim_raw = ans.get("ref_sim")
        ref_sim = float(ref_sim_raw) if isinstance(ref_sim_raw, (int, float)) else None
        ladder_class = ladder_per_row.get(idx)

        label = _classify_one(
            eval_row_index=idx,
            recall_k=recall_k,
            ref_sim=ref_sim,
            ladder_class=ladder_class,
            ref_sim_threshold=ref_sim_threshold,
            has_answer_eval=has_answer_eval,
        )
        counts[label] += 1

        q = str(trace_row.get("question_text") or "")
        if len(q) > 200:
            q = q[:197] + "..."

        per_query.append(
            {
                "eval_row_index": idx,
                "error_class": label,
                f"{primary}::recall@{k}": recall_k,
                "ref_sim": ref_sim,
                "ladder_class": ladder_class,
                "question_preview": q,
            }
        )

    n = len(per_query)
    fractions = {cls: cnt / max(1, n) for cls, cnt in counts.items()}

    examples: Dict[str, List[Dict[str, Any]]] = {}
    for cls in ("type1_retrieval", "type2_generation", "type0_ok", "unknown"):
        pool = [r for r in per_query if r["error_class"] == cls]
        examples[cls] = pool[:5]

    return {
        "schema": _SCHEMA,
        "trace": str(trace_path.as_posix()),
        "gold_jsonl": str(gold_path.as_posix()),
        "answer_jsonl": str(answer_path.as_posix()) if answer_path and answer_path.is_file() else None,
        "ladder_json": str(ladder_json.as_posix()) if ladder_json and ladder_json.is_file() else None,
        "primary": primary,
        "k": k,
        "ref_sim_threshold": ref_sim_threshold,
        "has_answer_eval": has_answer_eval,
        "n_queries": n,
        "counts": dict(counts),
        "fractions": fractions,
        "examples_top5_per_class": examples,
        "per_query": per_query,
        "rules_note": (
            "type1: recall@k=0 or ladder A/B; type2: recall@k>0 and ref_sim<threshold; "
            "type0_ok: recall@k>0 and (no answer-eval or ref_sim>=threshold)."
        ),
    }


def _write_md(report: Dict[str, Any], out_md: Path) -> None:
    lines = [
        "# RAG error classification (pilot)",
        "",
        f"- Trace: `{report.get('trace')}`",
        f"- Answer-eval: `{report.get('answer_jsonl') or '(none)'}`",
        f"- n_queries: **{report.get('n_queries')}**",
        f"- Rules: {report.get('rules_note')}",
        "",
        "## Counts",
        "",
        "| Class | Count | Fraction |",
        "|-------|------:|---------:|",
    ]
    for cls, cnt in sorted((report.get("counts") or {}).items()):
        frac = (report.get("fractions") or {}).get(cls, 0)
        lines.append(f"| `{cls}` | {cnt} | {frac:.1%} |")
    lines.append("")
    for cls, rows in (report.get("examples_top5_per_class") or {}).items():
        if not rows:
            continue
        lines.append(f"## Examples: `{cls}`")
        lines.append("")
        pk = report.get("primary")
        kk = report.get("k")
        rk = f"{pk}::recall@{kk}"
        for r in rows:
            lines.append(
                f"- idx={r.get('eval_row_index')} "
                f"R@{kk}={r.get(rk)} "
                f"ref_sim={r.get('ref_sim')} — {r.get('question_preview')}"
            )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--gold-jsonl", type=Path, default=Path("eval/datasets/eval_set_gold_v2.jsonl"))
    p.add_argument("--answer-jsonl", type=Path, default=None)
    p.add_argument("--ladder-json", type=Path, default=None, help="Optional; per_row[].class for A/B→type1")
    p.add_argument("--primary", default="substring")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ref-sim-threshold", type=float, default=0.55)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-per-query-jsonl", type=Path, default=None)
    args = p.parse_args()

    report = classify_rag_errors(
        trace_path=args.trace.resolve(),
        gold_path=args.gold_jsonl.resolve(),
        answer_path=args.answer_jsonl.resolve() if args.answer_jsonl else None,
        ladder_json=args.ladder_json.resolve() if args.ladder_json else None,
        primary=args.primary,
        k=args.k,
        ref_sim_threshold=args.ref_sim_threshold,
    )

    per_query = report.pop("per_query")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.out_per_query_jsonl:
        args.out_per_query_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_per_query_jsonl.open("w", encoding="utf-8") as sink:
            for row in per_query:
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.out_md:
        _write_md(report | {"per_query": per_query}, args.out_md)

    print(f"Wrote {args.out_json} (n={report.get('n_queries')}, counts={report.get('counts')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
