import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _gold_v2_path() -> Path:
    new = ROOT / "eval" / "datasets" / "eval_set_gold_v2.jsonl"
    return new if new.exists() else ROOT / "eval" / "eval_set_gold_v2.jsonl"


def _chunks_sample_path() -> Path:
    new = ROOT / "eval" / "datasets" / "chunks_sample.jsonl"
    return new if new.exists() else ROOT / "eval" / "fixtures" / "chunks_sample.jsonl"


def test_gold_v2_lines_parse():
    rows = [
        json.loads(line)
        for line in _gold_v2_path().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 10
    for r in rows:
        assert "question" in r and "references" in r and "answer_ref" in r
        assert isinstance(r["references"], list)


def test_gold_v2_references_substring_of_fixture_chunk():
    gold_path = _gold_v2_path()
    fix_path = _chunks_sample_path()
    docs = [
        json.loads(line)["document"]
        for line in fix_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold_lines = [line for line in gold_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(gold_lines) == len(docs)
    for i, line in enumerate(gold_lines):
        r = json.loads(line)
        doc = docs[i]
        for ref in r["references"]:
            assert ref in doc, (i, ref[:80])
