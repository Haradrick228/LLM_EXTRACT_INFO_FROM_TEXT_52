"""Unit tests for B3 run contract validation."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from eval.diagnostics.validate_research_run_contract import validate_b3_run_dir

_ROOT = Path(__file__).resolve().parents[2]
_PROTOCOL = _ROOT / "eval" / "config" / "final_research_protocol.yaml"
_CANONICAL = _ROOT / "eval" / "runs" / "20260513_114047_b3_full_corpus_rerank"


def test_canonical_b3_passes_with_semantic_waiver() -> None:
    if not _CANONICAL.is_dir():
        return
    protocol = yaml.safe_load(_PROTOCOL.read_text(encoding="utf-8"))
    ok, msgs = validate_b3_run_dir(_CANONICAL, protocol, allow_missing_semantic=True)
    assert ok, "\n".join(msgs)


def test_validate_flags_smoke_subset() -> None:
    protocol = yaml.safe_load(_PROTOCOL.read_text(encoding="utf-8"))
    smoke = _ROOT / "eval" / "runs" / "20260512_161557_b3_full_corpus_rerank"
    if not smoke.is_dir():
        return
    ok, _ = validate_b3_run_dir(smoke, protocol, allow_missing_semantic=True)
    assert not ok
