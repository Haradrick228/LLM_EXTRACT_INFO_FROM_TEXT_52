"""Выбрать худшие по primary mode строки из JSONL-трейса ``run_retrieval_eval``.

Использование::

    python -m eval.diagnostics.summarize_worst_retrieval_traces \\
        --trace eval/runs/<ts>_gold_on_full_corpus_bge_m3/trace.jsonl \\
        --primary substring \\
        --metric recall@10 \\
        --top 10 \\
        --out-md eval/RETRIEVAL_WORST_QUERIES_2026_05.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.rag.metrics.substring_match_hit import substring_match_hit


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--gold-jsonl", type=Path, default=Path("eval/datasets/eval_set_gold_v2.jsonl"))
    p.add_argument("--primary", default="substring", help="Ключ в metrics_by_mode")
    p.add_argument("--metric", default="recall@10")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out-md", type=Path, required=True)
    args = p.parse_args()

    gold = _load_gold_by_index(args.gold_jsonl.resolve())
    rows: List[Tuple[float, Dict[str, Any]]] = []
    with args.trace.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("record") != "query_trace":
                continue
            mm = o.get("metrics_by_mode") or {}
            block = mm.get(args.primary) or {}
            key = args.metric
            if key not in block:
                continue
            rows.append((float(block[key]), o))

    rows.sort(key=lambda x: x[0])
    worst = rows[: max(1, int(args.top))]

    lines: List[str] = [
        "# Worst retrieval queries (trace analysis)",
        "",
        f"- Trace: `{args.trace.as_posix()}`",
        f"- Gold: `{args.gold_jsonl.as_posix()}`",
        f"- Sort: `{args.primary}::{args.metric}` ascending (worst first), top **{len(worst)}**",
        "",
        "| idx | R@10 (sub) | question (trim) | ref in top1? | top1 snippet |",
        "|-----|------------|-----------------|--------------|--------------|",
    ]

    for r10, o in worst:
        idx = int(o.get("eval_row_index", -1))
        q = str(o.get("question_text", "")).replace("\n", " ").strip()
        if len(q) > 120:
            q = q[:117] + "..."
        g = gold.get(idx, {})
        refs = _refs_from_gold_line(g)
        ref0 = refs[0] if refs else ""
        st = o.get("stage_01_dense_chroma_retrieval") or {}
        cands = st.get("candidates_ordered_by_chroma") or []
        top1 = (cands[0].get("text") or "") if cands else ""
        hit = False
        if ref0 and top1:
            hit = substring_match_hit(top1, ref0)
        t1s = top1.replace("\n", " ").replace("|", "\\|")
        if len(t1s) > 100:
            t1s = t1s[:97] + "..."
        lines.append(f"| {idx} | {r10:.4f} | {q} | {hit} | {t1s} |")

    lines.extend(
        [
            "",
            "## Паттерны (эвристика)",
            "",
            "- Если **ref in top1?** часто `False`, но top1 выглядит тематически близко — вероятен **metric-limited** (substring / разрезание чанка).",
            "- Если top1 явно off-topic — **retriever-limited** или шум корпуса.",
            "",
        ]
    )

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
