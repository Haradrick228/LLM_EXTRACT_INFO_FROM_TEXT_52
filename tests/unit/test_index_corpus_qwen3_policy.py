"""Policy: full-corpus Qwen3-Embedding-* off в ``--all-st`` без PHASE_B_ALLOW_FULL_QWEN3=1."""
import pytest

from eval.indexing.index_corpus_from_jsonl import _apply_full_qwen3_index_policy


def test_explicit_qwen3_4b_allowed_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHASE_B_ALLOW_FULL_QWEN3", raising=False)
    sel = [{"base_id": "full_qwen3_embedding_4b"}]
    out = _apply_full_qwen3_index_policy(
        sel, explicit_base_id="full_qwen3_embedding_4b", all_st=False,
    )
    assert out == sel


def test_explicit_qwen3_8b_allowed_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHASE_B_ALLOW_FULL_QWEN3", raising=False)
    sel = [{"base_id": "full_qwen3_embedding_8b"}]
    out = _apply_full_qwen3_index_policy(
        sel, explicit_base_id="full_qwen3_embedding_8b", all_st=False,
    )
    assert out == sel


def test_all_st_drops_both_qwen3_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHASE_B_ALLOW_FULL_QWEN3", raising=False)
    sel = [
        {"base_id": "full_mpnet"},
        {"base_id": "full_qwen3_embedding_4b"},
        {"base_id": "full_qwen3_embedding_8b"},
    ]
    out = _apply_full_qwen3_index_policy(sel, explicit_base_id="", all_st=True)
    assert [b["base_id"] for b in out] == ["full_mpnet"]


def test_all_st_keeps_qwen3_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHASE_B_ALLOW_FULL_QWEN3", "1")
    sel = [
        {"base_id": "full_mpnet"},
        {"base_id": "full_qwen3_embedding_4b"},
        {"base_id": "full_qwen3_embedding_8b"},
    ]
    out = _apply_full_qwen3_index_policy(sel, explicit_base_id="", all_st=True)
    assert [b["base_id"] for b in out] == [
        "full_mpnet", "full_qwen3_embedding_4b", "full_qwen3_embedding_8b",
    ]


def test_no_all_st_keeps_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHASE_B_ALLOW_FULL_QWEN3", raising=False)
    sel = [
        {"base_id": "full_qwen3_embedding_4b"},
        {"base_id": "full_qwen3_embedding_8b"},
    ]
    out = _apply_full_qwen3_index_policy(sel, explicit_base_id="", all_st=False)
    assert out == sel
