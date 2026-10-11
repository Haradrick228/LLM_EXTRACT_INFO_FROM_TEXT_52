"""Unit tests for B3 matrix metric key discovery (no Chroma)."""
from __future__ import annotations

from eval.retrieval.run_retrieval_matrix import _metric_keys_from_payload


def test_metric_keys_skips_meta_and_non_numeric():
    payload = {
        "_meta": {"x": 1},
        "recall@10": 0.5,
        "token_f1::mrr": 0.2,
        "note": "str",
    }
    keys = _metric_keys_from_payload(payload)
    assert "recall@10" in keys
    assert "token_f1::mrr" in keys
    assert "_meta" not in keys
    assert "note" not in keys
    assert keys == sorted(keys)
