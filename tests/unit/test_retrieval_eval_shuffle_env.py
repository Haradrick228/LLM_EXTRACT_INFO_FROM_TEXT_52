"""RETRIEVAL_EVAL_SHUFFLE влияет на порядок строк в load_eval (детерминизм матрицы B3)."""
import json
import os
from pathlib import Path

from eval.retrieval.run_retrieval_eval import load_eval


def test_load_eval_shuffle_false_stable_order(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    p.write_text(
        '{"question":"a","references":[]}\n{"question":"b","references":[]}\n',
        encoding="utf-8",
    )
    prev = os.environ.get("RETRIEVAL_EVAL_SHUFFLE")
    try:
        os.environ["RETRIEVAL_EVAL_SHUFFLE"] = "0"
        r1 = load_eval(p, shuffle=True)
        r2 = load_eval(p, shuffle=True)
        assert [x["question"] for x in r1] == ["a", "b"]
        assert [x["question"] for x in r2] == ["a", "b"]
    finally:
        if prev is None:
            os.environ.pop("RETRIEVAL_EVAL_SHUFFLE", None)
        else:
            os.environ["RETRIEVAL_EVAL_SHUFFLE"] = prev


def test_load_eval_shuffle_true_can_reorder(tmp_path: Path) -> None:
    p = tmp_path / "e2.jsonl"
    p.write_text(
        '{"question":"a","references":[]}\n{"question":"b","references":[]}\n',
        encoding="utf-8",
    )
    prev = os.environ.get("RETRIEVAL_EVAL_SHUFFLE")
    try:
        os.environ.pop("RETRIEVAL_EVAL_SHUFFLE", None)
        seen = set()
        for _ in range(200):
            rows = load_eval(p, shuffle=True)
            seen.add(tuple(x["question"] for x in rows))
        assert len(seen) >= 2
    finally:
        if prev is not None:
            os.environ["RETRIEVAL_EVAL_SHUFFLE"] = prev
