"""Реестр баз фазы B: шесть записей, уникальные base_id и коллекции."""
from __future__ import annotations

from pathlib import Path

from eval.experiments.phase_b_collections_registry import list_bases, load_registry


def test_embedding_bases_yaml_has_six_ordered_entries():
    root = Path(__file__).resolve().parents[2]
    yaml_path = root / "eval" / "config" / "embedding_bases_fixture.yaml"
    if not yaml_path.exists():
        yaml_path = root / "eval" / "phase_b" / "embedding_bases.yaml"
    data = load_registry(yaml_path)
    bases = list_bases(data)
    assert len(bases) == 6
    ids = [str(b["base_id"]) for b in bases]
    assert len(ids) == len(set(ids))
    cols = [str(b["collection"]) for b in bases]
    assert len(cols) == len(set(cols))
    assert bases[0]["base_id"] == "gold_mpnet_fixture"
    assert bases[-1]["base_id"] == "gold_giga_embeddings_instruct"
    assert any(b.get("heavy_index_last") for b in bases)
