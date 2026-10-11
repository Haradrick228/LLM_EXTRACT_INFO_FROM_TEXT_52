"""Probe semantic cosine scores between top-1 chunk and gold reference (calibration aid).

Samples ``query_trace`` rows from a retrieval trace, pairs the first retrieved chunk with the
primary reference from ``--gold-jsonl``, and reports score distribution from
``build_semantic_similarity_scorer`` (same path as ``--metric-modes semantic`` in retrieval eval).

Does **not** auto-tune production threshold; use percentiles + [eval/METRICS_LADDER.md](../METRICS_LADDER.md).

Usage::

    python -m eval.diagnostics.semantic_threshold_probe \\
        --trace eval/runs/<id>/trace.jsonl \\
        --gold-jsonl eval/datasets/eval_set_gold_v2.jsonl \\
        --semantic-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \\
        --max-rows 25 \\
        --out-json eval/runs/<id>/semantic_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]

from core.rag.metrics.semantic_similarity_hit import build_semantic_similarity_scorer
from core.rag.metrics.substring_match_hit import substring_match_hit


def _load_gold_by_index(path: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            out[i] = json.loads(line)
    return out


def _refs_from_gold_line(row: Dict[str, Any]) -> List[str]:
    raw = row.get("references")
    out: List[str] = []
    if isinstance(raw, list):
        out = [str(x).strip() for x in raw if str(x).strip()]
    if out:
        return out
    rq = (row.get("reference_quote") or "").strip()
    return [rq] if rq else []


def _iter_query_traces(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("record") == "query_trace":
                yield o


def _top_chunk_text(o: Dict[str, Any]) -> str:
    st = o.get("stage_01_dense_chroma_retrieval") or {}
    cands = st.get("candidates_ordered_by_chroma") or []
    if not cands:
        return ""
    first = cands[0]
    if isinstance(first, dict):
        return str(first.get("text") or "")
    return ""


def _percentiles(sorted_vals: List[float], ps: Tuple[int, ...]) -> Dict[str, float]:
    if not sorted_vals:
        return {f"p{p}": 0.0 for p in ps}
    n = len(sorted_vals)
    out: Dict[str, float] = {}
    for p in ps:
        idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
        out[f"p{p}"] = float(sorted_vals[idx])
    return out


def run_probe(
    *,
    trace_path: Path,
    gold_path: Path,
    semantic_model: str,
    max_rows: int,
    device: Optional[str],
) -> Dict[str, Any]:
    gold = _load_gold_by_index(gold_path)
    scorer = build_semantic_similarity_scorer(semantic_model, threshold=0.62, device=device)
    scores: List[float] = []
    scores_sub_hit: List[float] = []
    scores_sub_miss: List[float] = []
    rows_used = 0

    for o in _iter_query_traces(trace_path):
        if max_rows > 0 and rows_used >= max_rows:
            break
        idx = int(o.get("eval_row_index", -1))
        refs = _refs_from_gold_line(gold.get(idx, {}))
        if not refs:
            continue
        ref0 = refs[0]
        chunk = _top_chunk_text(o)
        if not chunk:
            continue
        s = float(scorer.score_fn(chunk, ref0))
        scores.append(s)
        if substring_match_hit(chunk, ref0):
            scores_sub_hit.append(s)
        else:
            scores_sub_miss.append(s)
        rows_used += 1

    scores.sort()
    sh = sorted(scores_sub_hit)
    sm = sorted(scores_sub_miss)
    return {
        "schema": "semantic_threshold_probe_v1",
        "trace": str(trace_path.as_posix()),
        "gold_jsonl": str(gold_path.as_posix()),
        "semantic_model": semantic_model,
        "scorer_backend": scorer.backend,
        "default_threshold_in_eval": float(os.getenv("RETRIEVAL_SEMANTIC_THRESHOLD", "0.62") or 0.62),
        "n_sampled_rows": len(scores),
        "percentiles_all": _percentiles(scores, (10, 50, 90)),
        "percentiles_substring_hit": _percentiles(sh, (50, 90)) if sh else {},
        "percentiles_substring_miss": _percentiles(sm, (50, 90)) if sm else {},
        "note": "Tune RETRIEVAL_SEMANTIC_THRESHOLD so substring-miss pairs with plausible relevance "
        "score above threshold sparingly; compare percentiles_substring_hit vs _miss.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--gold-jsonl", type=Path, default=_ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl")
    p.add_argument(
        "--semantic-model",
        default=os.getenv(
            "RETRIEVAL_SEMANTIC_MODEL",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ),
    )
    p.add_argument("--max-rows", type=int, default=25, help="Max query_trace rows to score (0 = no limit).")
    p.add_argument("--device", default=os.getenv("RERANK_DEVICE"), help="Optional torch device for ST.")
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    max_rows = int(args.max_rows)
    report = run_probe(
        trace_path=args.trace.resolve(),
        gold_path=args.gold_jsonl.resolve(),
        semantic_model=str(args.semantic_model),
        max_rows=max_rows,
        device=(str(args.device).strip() or None) if args.device else None,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Wrote", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
