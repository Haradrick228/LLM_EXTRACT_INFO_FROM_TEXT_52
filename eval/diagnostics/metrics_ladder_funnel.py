"""Funnel metrics for retrieval eval: corpus oracle (class A), pool oracle (B), rank (C), hit@k (D).

Reads ``run_retrieval_eval`` JSONL traces (``record=query_trace``) plus gold JSONL aligned by
``eval_row_index``. Optionally scans a corpus JSONL (``id``, ``document``) to estimate whether
any reference text appears in the indexed corpus (class A vs not-A).

Usage::

    python -m eval.diagnostics.metrics_ladder_funnel \\
        --trace eval/runs/<id>/trace.jsonl \\
        --gold-jsonl eval/datasets/eval_set_gold_v2.jsonl \\
        --corpus-jsonl DATA/runs/corpus_full_export_20260512/chunks_from_chroma.jsonl \\
        --k-primary 10 \\
        --out-json eval/runs/<id>/metrics_ladder.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from core.rag.metrics.ranking_aggregates import (
    mean_reciprocal_rank as _mrr_pluggable,
    recall_at_k as _recall_pluggable,
)
from core.rag.metrics.substring_match_hit import substring_match_hit
from core.rag.metrics.token_overlap_f1_hit import token_overlap_f1_hit

_ROOT = Path(__file__).resolve().parents[2]


def _refs_from_gold_line(row: Dict[str, Any]) -> List[str]:
    raw = row.get("references")
    out: List[str] = []
    if isinstance(raw, list):
        out = [str(x).strip() for x in raw if str(x).strip()]
    if out:
        return out
    rq = (row.get("reference_quote") or "").strip()
    return [rq] if rq else []


def _load_gold_by_index(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            out[i] = json.loads(line)
    return out


def _wilson_interval(successes: int, n: int, *, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt(max(0.0, (p * (1.0 - p) / n + z * z / (4.0 * n * n)))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _hit_fn_for_name(name: str) -> Callable[[str, str], bool]:
    if name == "substring":
        return substring_match_hit
    if name == "token_f1":
        return token_overlap_f1_hit
    raise ValueError(f"unknown hit_fn name: {name!r} (use substring|token_f1)")


def _docs_from_trace_stage(stage: Dict[str, Any], key: str) -> List[str]:
    cands = stage.get(key) or []
    out: List[str] = []
    for c in cands:
        if isinstance(c, dict):
            out.append(str(c.get("text") or ""))
    return out


def _ambiguous_rate(stage2: Dict[str, Any]) -> float:
    cands = stage2.get("candidates_ordered_after_rerank") or []
    if not cands:
        return 0.0
    amb = 0
    for c in cands:
        if isinstance(c, dict) and c.get("match_ambiguous"):
            amb += 1
    return amb / max(1, len(cands))


def _classify_row(
    *,
    in_corpus: Optional[bool],
    pre_docs: List[str],
    post_docs: List[str],
    refs: List[str],
    hit_fn: Callable[[str, str], bool],
    k_primary: int,
) -> str:
    """Return A|B|C|D|A_skip."""
    if not refs:
        return "A_skip"
    if in_corpus is False:
        return "A"

    oracle_pre = any(hit_fn(d, r) for d in pre_docs for r in refs)
    if not oracle_pre:
        return "B"

    hit_post = _recall_pluggable(post_docs, refs, k_primary, hit_fn=hit_fn) > 0.0
    if hit_post:
        return "D"
    return "C"


def stream_corpus_contains_ref(
    corpus_path: Path,
    refs_by_row: Dict[int, List[str]],
    hit_fn: Callable[[str, str], bool],
    *,
    max_corpus_lines: Optional[int] = None,
) -> Dict[int, bool]:
    """For each row index, True if any ref hits any corpus document (single streaming pass)."""
    found: Dict[int, bool] = {i: False for i in refs_by_row}
    pending = set(i for i, rs in refs_by_row.items() if rs)
    if not pending:
        return found

    n_lines = 0
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n_lines += 1
            if max_corpus_lines is not None and n_lines > max_corpus_lines:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc = str(row.get("document") or "")
            if not doc:
                continue
            to_remove: List[int] = []
            for i in pending:
                for ref in refs_by_row.get(i) or []:
                    if hit_fn(doc, ref):
                        found[i] = True
                        to_remove.append(i)
                        break
            for i in to_remove:
                pending.discard(i)
            if not pending:
                break
    return found


def _iter_query_traces(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("record") == "query_trace":
                yield o


def _mean(xs: List[float]) -> float:
    return sum(xs) / max(1, len(xs))


def build_funnel_report(
    *,
    trace_path: Path,
    gold_path: Path,
    corpus_path: Optional[Path],
    hit_name: str,
    k_primary: int,
    max_corpus_lines: Optional[int] = None,
) -> Dict[str, Any]:
    hit_fn = _hit_fn_for_name(hit_name)
    gold = _load_gold_by_index(gold_path)
    refs_by_row = {i: _refs_from_gold_line(row) for i, row in gold.items()}

    in_corpus: Optional[Dict[int, bool]] = None
    if corpus_path is not None and corpus_path.is_file():
        in_corpus = stream_corpus_contains_ref(
            corpus_path,
            refs_by_row,
            hit_fn,
            max_corpus_lines=max_corpus_lines,
        )

    counts: Dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "A_skip": 0}
    hit_binaries_post: List[int] = []
    cond_rows_recall: List[float] = []
    cond_rows_mrr: List[float] = []
    rerank_traces = 0
    amb_rates: List[float] = []
    d_mrr: List[float] = []
    d_recall: List[float] = []

    for o in _iter_query_traces(trace_path):
        idx = int(o.get("eval_row_index", -1))
        g = gold.get(idx, {})
        refs = _refs_from_gold_line(g)
        st1 = o.get("stage_01_dense_chroma_retrieval") or {}
        pre_docs = _docs_from_trace_stage(st1, "candidates_ordered_by_chroma")
        st2 = o.get("stage_02_rerank") or {}
        post_docs = _docs_from_trace_stage(st2, "candidates_ordered_after_rerank")
        if not post_docs:
            post_docs = list(pre_docs)
        has_rerank = bool(st2.get("candidates_ordered_after_rerank"))
        if has_rerank:
            rerank_traces += 1
            amb_rates.append(_ambiguous_rate(st2))

        ic: Optional[bool] = None
        if in_corpus is not None:
            ic = bool(in_corpus.get(idx, False))

        clazz = _classify_row(
            in_corpus=ic,
            pre_docs=pre_docs,
            post_docs=post_docs,
            refs=refs,
            hit_fn=hit_fn,
            k_primary=k_primary,
        )
        counts[clazz] = counts.get(clazz, 0) + 1

        oracle_pre = any(hit_fn(d, r) for d in pre_docs for r in refs) if refs else False
        r_pre = _recall_pluggable(pre_docs, refs, k_primary, hit_fn=hit_fn) if refs else 0.0
        r_post = _recall_pluggable(post_docs, refs, k_primary, hit_fn=hit_fn) if refs else 0.0
        m_pre = _mrr_pluggable(pre_docs, refs, hit_fn=hit_fn) if refs else 0.0
        m_post = _mrr_pluggable(post_docs, refs, hit_fn=hit_fn) if refs else 0.0

        hit_binaries_post.append(1 if r_post > 0.0 else 0)

        if oracle_pre:
            cond_rows_recall.append(r_post)
            cond_rows_mrr.append(m_post)

        if has_rerank and refs:
            d_mrr.append(m_post - m_pre)
            d_recall.append(r_post - r_pre)

    n = len(hit_binaries_post)
    successes = sum(hit_binaries_post)
    lo, hi = _wilson_interval(successes, n)

    return {
        "schema": "metrics_ladder_funnel_v1",
        "trace": str(trace_path.as_posix()),
        "gold_jsonl": str(gold_path.as_posix()),
        "corpus_jsonl": str(corpus_path.as_posix()) if corpus_path else None,
        "hit_fn": hit_name,
        "k_primary": k_primary,
        "n_queries": n,
        "class_counts": counts,
        "class_fractions": {k: v / max(1, n) for k, v in counts.items()},
        "wilson_hit_at_k_post": {"k": k_primary, "successes": successes, "lo": lo, "hi": hi},
        "conditional_given_oracle_pool": {
            "n": len(cond_rows_recall),
            "mean_recall_at_k_post": _mean(cond_rows_recall),
            "mean_mrr_post": _mean(cond_rows_mrr),
        },
        "rerank": {
            "n_traces_with_stage2": rerank_traces,
            "mean_delta_mrr": _mean(d_mrr) if d_mrr else None,
            "mean_delta_recall_at_k": _mean(d_recall) if d_recall else None,
            "mean_ambiguous_rate": _mean(amb_rates) if amb_rates else None,
        },
    }


def _write_md_summary(report: Dict[str, Any], out_md: Path) -> None:
    fc = report.get("class_fractions") or {}
    lines = [
        "# Metrics ladder (funnel)",
        "",
        f"- Trace: `{report.get('trace')}`",
        f"- Gold: `{report.get('gold_jsonl')}`",
        f"- Corpus oracle file: `{report.get('corpus_jsonl') or '(none — class A not computed)'}`",
        f"- Hit: **{report.get('hit_fn')}**, k_primary=**{report.get('k_primary')}**",
        f"- Queries: **{report.get('n_queries')}**",
        "",
        "## Class mix (A=gold not in corpus text, B=not in pool, C=in pool but miss@k, D=hit@k)",
        "",
        "| class | count | fraction |",
        "|-------|------:|---------:|",
    ]
    cc = report.get("class_counts") or {}
    for k in ("A", "B", "C", "D", "A_skip"):
        if k in cc:
            lines.append(f"| {k} | {cc[k]} | {fc.get(k, 0):.4f} |")
    w = report.get("wilson_hit_at_k_post") or {}
    lines.extend(
        [
            "",
            "## Wilson 95% CI for hit@k (post-pipeline, binary per query)",
            "",
            f"- k={w.get('k')}, successes={w.get('successes')}, CI≈[{w.get('lo'):.4f}, {w.get('hi'):.4f}]",
            "",
            "## Conditional on oracle_pool (gold hit somewhere in pre-fetch_k)",
            "",
            json.dumps(report.get("conditional_given_oracle_pool") or {}, ensure_ascii=False, indent=2),
            "",
            "## Rerank deltas (only traces with stage_02)",
            "",
            json.dumps(report.get("rerank") or {}, ensure_ascii=False, indent=2),
            "",
        ]
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--gold-jsonl", type=Path, default=_ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl")
    p.add_argument("--corpus-jsonl", type=Path, default=None, help="Optional corpus export for class A.")
    p.add_argument(
        "--max-corpus-lines",
        type=int,
        default=0,
        help="0 = read entire corpus file; else cap lines (smoke only — breaks class A if set too low).",
    )
    p.add_argument("--hit-fn", choices=("substring", "token_f1"), default="substring")
    p.add_argument("--k-primary", type=int, default=10)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, default=None)
    args = p.parse_args()

    cap = args.max_corpus_lines
    max_lines = None if cap == 0 else cap

    report = build_funnel_report(
        trace_path=args.trace.resolve(),
        gold_path=args.gold_jsonl.resolve(),
        corpus_path=args.corpus_jsonl.resolve() if args.corpus_jsonl else None,
        hit_name=args.hit_fn,
        k_primary=int(args.k_primary),
        max_corpus_lines=max_lines,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")
    if args.out_md:
        _write_md_summary(report, args.out_md.resolve())
        print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
