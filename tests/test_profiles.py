import os

import pytest

from common.rag_profile_env import apply_profile, list_profile_names


def test_list_matrix_profiles():
    names = list_profile_names(matrix_only=True)
    assert "rerank_jina_v3" in names
    assert "ollama_answer_eval" not in names


def test_apply_profile_sets_rerank(monkeypatch):
    base = dict(os.environ)
    try:
        apply_profile("rerank_jina_v3")
        assert os.getenv("RERANK_MODEL") == "jinaai/jina-reranker-v3"
        assert os.getenv("RERANK_MODE") == "cross"
    finally:
        os.environ.clear()
        os.environ.update(base)


def test_apply_unknown_raises():
    with pytest.raises(KeyError):
        apply_profile("no_such_profile_xyz")
