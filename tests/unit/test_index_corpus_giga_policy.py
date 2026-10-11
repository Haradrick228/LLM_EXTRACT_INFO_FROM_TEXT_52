"""Policy: full-corpus Giga index off в ``--all-st`` без ``PHASE_B_ALLOW_FULL_GIGA=1``."""
import pytest

from eval.indexing.index_corpus_from_jsonl import _apply_full_giga_index_policy


def test_explicit_full_giga_allowed_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Явный ``--base-id full_giga_embeddings_instruct`` всегда возвращает выборку как есть."""
    monkeypatch.delenv("PHASE_B_ALLOW_FULL_GIGA", raising=False)
    sel = [{"base_id": "full_giga_embeddings_instruct"}]
    out = _apply_full_giga_index_policy(
        sel,
        explicit_base_id="full_giga_embeddings_instruct",
        all_st=False,
    )
    assert out == sel


def test_explicit_full_giga_allowed_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHASE_B_ALLOW_FULL_GIGA", "1")
    sel = [{"base_id": "full_giga_embeddings_instruct"}]
    out = _apply_full_giga_index_policy(
        sel,
        explicit_base_id="full_giga_embeddings_instruct",
        all_st=True,
    )
    assert out == sel


def test_all_st_drops_giga_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHASE_B_ALLOW_FULL_GIGA", raising=False)
    sel = [
        {"base_id": "full_mpnet"},
        {"base_id": "full_giga_embeddings_instruct"},
    ]
    out = _apply_full_giga_index_policy(sel, explicit_base_id="", all_st=True)
    assert [b["base_id"] for b in out] == ["full_mpnet"]


def test_all_st_keeps_giga_with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHASE_B_ALLOW_FULL_GIGA", "1")
    sel = [
        {"base_id": "full_mpnet"},
        {"base_id": "full_giga_embeddings_instruct"},
    ]
    out = _apply_full_giga_index_policy(sel, explicit_base_id="", all_st=True)
    assert [b["base_id"] for b in out] == ["full_mpnet", "full_giga_embeddings_instruct"]
