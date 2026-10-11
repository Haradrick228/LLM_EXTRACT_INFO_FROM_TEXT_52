import json
import tempfile
from pathlib import Path

from eval.datasets.load_gold_fixture_chunks import load_fixture_batches


def test_load_fixture_batches_clean_text_strips_control_chars():
    d = Path(tempfile.mkdtemp())
    p = d / "rows.jsonl"
    p.write_text(
        json.dumps({"id": "x1", "document": "pre\x00post", "metadata": {"k": 1}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ids, docs, metas = load_fixture_batches(p, apply_clean_text=True)
    assert ids == ["x1"]
    assert docs == ["pre post"]
    assert metas == [{"k": 1}]


def test_load_fixture_batches_raw_when_disabled():
    d = Path(tempfile.mkdtemp())
    p = d / "rows.jsonl"
    raw = "pre\x00post"
    p.write_text(
        json.dumps({"id": "x1", "document": raw, "metadata": {}}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ids, docs, _ = load_fixture_batches(p, apply_clean_text=False)
    assert docs == [raw]
