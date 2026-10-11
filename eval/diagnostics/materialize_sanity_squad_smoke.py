"""Materialize tiny SQuAD-based corpus + gold for external sanity (Layer 8).

Writes:
  DATA/runs/sanity_external_squad_smoke/corpus.jsonl  — id, document, metadata
  DATA/runs/sanity_external_squad_smoke/gold.jsonl     — question, references (verbatim answer)

License: SQuAD (rajpurkar/squad) — CC BY-SA 4.0.

Usage::

    python -m eval.diagnostics.materialize_sanity_squad_smoke --n 80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=80, help="Number of train examples.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "DATA" / "runs" / "sanity_external_squad_smoke",
    )
    args = p.parse_args()
    try:
        from datasets import load_dataset
    except Exception as exc:
        print("pip install datasets", exc, flush=True)
        return 2

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    corp_path = out_dir / "corpus.jsonl"
    gold_path = out_dir / "gold.jsonl"

    split = f"train[:{int(args.n)}]"
    ds = load_dataset("rajpurkar/squad", split=split)

    n_written = 0
    with corp_path.open("w", encoding="utf-8") as fc, gold_path.open("w", encoding="utf-8") as fg:
        for i, row in enumerate(ds):
            ctx = str(row.get("context") or "").strip()
            q = str(row.get("question") or "").strip()
            ans = str(row.get("answers", {}).get("text", [""])[0] or "").strip()
            if not ctx or not q or not ans:
                continue
            if ans not in ctx:
                continue
            cid = f"squad_train_{i}"
            fc.write(
                json.dumps(
                    {"id": cid, "document": ctx, "metadata": {"source": "squad", "idx": i}},
                    ensure_ascii=False,
                )
                + "\n"
            )
            fg.write(
                json.dumps(
                    {
                        "question": q,
                        "references": [ans],
                        "reference_quote": ans,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    (out_dir / "README.txt").write_text(
        "SQuAD smoke corpus. License: CC BY-SA 4.0 (rajpurkar/squad).\n"
        f"rows={n_written}\n",
        encoding="utf-8",
    )
    print(f"Wrote {n_written} rows to {corp_path} and {gold_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
