from eval.validate_gold_v2 import validate


def test_gold_v2_validate_passes():
    errs, summary = validate()
    assert not errs, "\n".join(errs)
    assert summary["fixture_rows"] == summary["gold_rows"]
    assert summary["fixture_rows"] >= 10
