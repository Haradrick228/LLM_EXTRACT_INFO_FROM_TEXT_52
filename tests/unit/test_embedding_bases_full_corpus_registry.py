"""embedding_bases_full_corpus.yaml: F6 добавил 3 новые базы (Giga, Qwen3-4B, Qwen3-8B)."""
from __future__ import annotations

from pathlib import Path

from eval.experiments.phase_b_collections_registry import list_bases, load_registry


def _yaml() -> Path:
    root = Path(__file__).resolve().parents[2]
    p = root / "eval" / "config" / "embedding_bases_full_corpus.yaml"
    if not p.exists():
        p = root / "eval" / "phase_b" / "embedding_bases_full_corpus.yaml"
    return p


def test_contains_baseline_five_plus_default_clean():
    data = load_registry(_yaml())
    ids = [str(b["base_id"]) for b in list_bases(data)]
    for required in (
        "full_mpnet",
        "full_bge_m3",
        "full_frida",
        "full_e5_large_instruct",
        "full_jina_embeddings_v3",
        "full_default_clean",
    ):
        assert required in ids, f"missing {required!r} in registry"


def test_contains_f6_new_embedders():
    data = load_registry(_yaml())
    ids = [str(b["base_id"]) for b in list_bases(data)]
    for new in (
        "full_giga_embeddings_instruct",
        "full_qwen3_embedding_4b",
        "full_qwen3_embedding_8b",
    ):
        assert new in ids, f"F6 base {new!r} not in registry"


def test_new_embedders_have_trust_remote_code():
    data = load_registry(_yaml())
    by_id = {str(b["base_id"]): b for b in list_bases(data)}
    for bid in ("full_giga_embeddings_instruct", "full_qwen3_embedding_4b", "full_qwen3_embedding_8b"):
        st_kw = (by_id[bid].get("st_model_kwargs") or {})
        assert bool(st_kw.get("trust_remote_code")) is True, f"{bid} missing trust_remote_code"


def test_collection_names_unique():
    data = load_registry(_yaml())
    cols = [str(b["collection"]) for b in list_bases(data)]
    assert len(cols) == len(set(cols)), f"duplicate collections: {cols}"
