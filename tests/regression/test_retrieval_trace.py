"""JSONL trace from eval.retrieval.run_retrieval_eval (dense Chroma + rerank stages)."""
import json
from pathlib import Path

from eval.retrieval.run_retrieval_eval import evaluate, load_eval


def test_load_eval_shuffle_false_preserves_file_order(tmp_path: Path) -> None:
    p = tmp_path / "e.jsonl"
    p.write_text(
        '{"question":"first","references":[]}\n{"question":"second","references":[]}\n',
        encoding="utf-8",
    )
    rows = load_eval(p, shuffle=False)
    assert [r["question"] for r in rows] == ["first", "second"]


def test_evaluate_trace_jsonl_query_and_rerank_stages(tmp_path: Path) -> None:
    class _Col:
        def query(self, query_texts, n_results, include):
            assert query_texts == ["q?"]
            assert n_results == 2
            assert "documents" in include
            return {
                "documents": [["doc_a", "doc_b"]],
                "distances": [[0.5, 0.6]],
                "metadatas": [[{"id": "id_a"}, {"id": "id_b"}]],
            }

    class _Rerank:
        def predict(self, pairs):
            import numpy as np

            # CrossEncoder returns ndarray; rerank() calls .tolist()
            return np.array([1.0, 3.0], dtype=np.float32)

    out = tmp_path / "trace.jsonl"
    sink = out.open("w", encoding="utf-8")
    meta = {
        "trace_profile_label": "test_profile",
        "chroma_collection": "corpus_x",
        "embed_model": "embed/model",
        "rerank_model": "rerank/model",
        "rerank_device": "cpu",
    }
    dataset = [{"question": "q?", "references": ["doc_b"]}]
    evaluate(
        _Col(),
        dataset,
        [5],
        fetch_k=2,
        rerank_model=_Rerank(),
        rerank_mode="cross",
        trace_sink=sink,
        trace_doc_chars=50,
        trace_meta=meta,
    )
    sink.close()
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    qt = [x for x in lines if x.get("record") == "query_trace"]
    assert len(qt) == 1
    row = qt[0]
    assert row["trace_profile_label"] == "test_profile"
    assert row["stage_01_dense_chroma_retrieval"]["n_candidates_returned"] == 2
    assert row["stage_01_dense_chroma_retrieval"]["candidates_ordered_by_chroma"][0]["chunk_id"] == "id_a"
    assert row["stage_02_rerank"]["candidates_ordered_after_rerank"][0]["rank_after_rerank"] == 1
    assert row["stage_02_rerank"]["candidates_ordered_after_rerank"][0]["reranker_score"] == 3.0
    assert row["stage_02_rerank"]["candidates_ordered_after_rerank"][0]["rank_after_chroma_before_rerank"] == 2
