import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gold_v2_lines_parse():
    path = ROOT / "eval" / "eval_set_gold_v2.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 10
    for r in rows:
        assert "question" in r and "references" in r and "answer_ref" in r
        assert isinstance(r["references"], list)


def test_gold_v2_references_substring_of_fixture_chunk():
    gold_path = ROOT / "eval" / "eval_set_gold_v2.jsonl"
    fix_path = ROOT / "eval" / "fixtures" / "chunks_sample.jsonl"
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
