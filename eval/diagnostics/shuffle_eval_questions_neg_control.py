"""Negative control: misalign questions and gold rows (same corpus, wrong Q↔ref pairing).

Writes a new JSONL suitable for ``run_retrieval_eval`` where each line keeps ``references``
from row *i* but takes ``question`` from row *i+1* (circular). Expected retrieval metrics
collapse toward random / near-zero on a large corpus when substring match is used.

Usage::

    python -m eval.diagnostics.shuffle_eval_questions_neg_control \\
        --in-jsonl eval/datasets/eval_set_gold_v2.jsonl \\
        --out-jsonl eval/runs/_neg_control/shuffled_questions_gold_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-jsonl", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    args = p.parse_args()

    rows: List[Dict[str, Any]] = []
    with args.in_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    if len(rows) < 2:
        print("need at least 2 rows", file=__import__("sys").stderr)
        return 2

    out: List[Dict[str, Any]] = []
    n = len(rows)
    for i, row in enumerate(rows):
        nxt = rows[(i + 1) % n]
        new_row = dict(row)
        new_row["question"] = nxt.get("question", "")
        new_row["_neg_control"] = "question_from_next_row_circular"
        out.append(new_row)

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as fo:
        for r in out:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} rows to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
