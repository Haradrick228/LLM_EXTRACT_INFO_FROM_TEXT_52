from pathlib import Path

from eval.retrieval.run_retrieval_matrix import _scenario_rows

ROOT = Path(__file__).resolve().parents[2]


def test_scenario_rows_from_full_corpus_registry():
    reg = ROOT / "eval" / "config" / "embedding_bases_full_corpus.yaml"
    if not reg.exists():
        reg = ROOT / "eval" / "phase_b" / "embedding_bases_full_corpus.yaml"
    rows = _scenario_rows(reg)
    by_id = {r[0]: r for r in rows}
    assert len(by_id) >= 6
    assert by_id["full_mpnet"][1] == "corpus_phase_b_mpnet"
    assert "paraphrase-multilingual-mpnet" in by_id["full_mpnet"][2]
    assert by_id["full_jina_embeddings_v3"][1] == "corpus_phase_b_jina_embeddings_v3"
    assert by_id["full_default_clean"][1] == "default_clean"
    assert by_id["full_default_clean"][3] == "default"
