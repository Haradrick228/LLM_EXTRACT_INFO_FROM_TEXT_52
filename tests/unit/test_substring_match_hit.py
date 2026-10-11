"""Substring match hit: подстрока в любую сторону, lower-case, пустые → False."""
from core.rag.metrics.substring_match_hit import substring_match_hit


def test_ref_subset_of_doc():
    assert substring_match_hit("целеполагание на уроке: критерии успеха", "критерии успеха")


def test_doc_subset_of_ref():
    assert substring_match_hit("критерии успеха", "целеполагание на уроке: критерии успеха")


def test_case_insensitive():
    assert substring_match_hit("Критерии Успеха", "критерии УСПЕХА")


def test_no_overlap():
    assert not substring_match_hit("оценивание по рубрикам", "проектная деятельность")


def test_empty():
    assert not substring_match_hit("", "что-то")
    assert not substring_match_hit("что-то", "")
    assert not substring_match_hit("", "")
