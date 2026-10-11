"""Source-/chunk-id match: совпадение хотя бы одного ключа."""
from core.rag.metrics.source_id_match_hit import source_id_match_hit


def test_source_match():
    retrieved = {"source": "edu/curriculum_2023.pdf", "chunk_id": "abc"}
    assert source_id_match_hit(
        retrieved,
        references_source=["edu/curriculum_2023.pdf"],
        references_chunk_id=[],
    )


def test_chunk_id_match():
    retrieved = {"source": "irrelevant.pdf", "chunk_id": "ABCDEF"}
    assert source_id_match_hit(
        retrieved,
        references_source=[],
        references_chunk_id=["abcdef"],
    )


def test_case_insensitive():
    retrieved = {"source": "DOC.PDF", "chunk_id": ""}
    assert source_id_match_hit(retrieved, references_source=["doc.pdf"])


def test_no_match():
    retrieved = {"source": "x.pdf", "chunk_id": "y"}
    assert not source_id_match_hit(retrieved, references_source=["z.pdf"], references_chunk_id=["w"])


def test_empty_inputs():
    assert not source_id_match_hit({}, references_source=["a"], references_chunk_id=["b"])
    assert not source_id_match_hit({"source": "a"}, references_source=[], references_chunk_id=[])
