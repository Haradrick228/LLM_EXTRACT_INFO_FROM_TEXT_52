from core.rag.rerank_registry import load_reranker


def test_load_empty_returns_none():
    m, kind = load_reranker("", "cross")
    assert m is None
    assert kind == "none"
