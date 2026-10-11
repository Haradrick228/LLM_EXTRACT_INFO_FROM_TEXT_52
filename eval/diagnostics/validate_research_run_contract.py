"""Validate B3 / retrieval run artifacts against ``final_research_protocol.yaml``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_protocol(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid protocol yaml: {path}")
    return data


def _as_modes(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def _check_meta(meta: Dict[str, Any], required: List[str], *, ctx: str) -> List[str]:
    errs: List[str] = []
    for key in required:
        if key not in meta:
            errs.append(f"{ctx}: missing _meta.{key}")
    return errs


def validate_b3_run_dir(
    run_dir: Path,
    protocol: Dict[str, Any],
    *,
    allow_missing_semantic: bool = False,
) -> Tuple[bool, List[str]]:
    """Return (ok, messages)."""
    msgs: List[str] = []
    ok = True

    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return False, [f"missing summary.json in {run_dir}"]

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest_path = run_dir / "matrix_manifest.json"
    manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("metric_modes", "semantic_threshold", "semantic_model", "eval_scenario_type"):
            if key not in summary and key in manifest:
                summary[key] = manifest[key]
    b3 = protocol.get("retrieval_matrix_b3") or {}
    corpus = protocol.get("corpus") or {}
    req_summary = list(protocol.get("required_summary_fields") or [])
    req_meta = list(protocol.get("required_retrieval_meta_fields") or [])

    if not summary.get("eval_scenario_type"):
        summary["eval_scenario_type"] = b3.get("eval_scenario_type", "full_corpus")
    if not summary.get("metric_modes"):
        summary["metric_modes"] = b3.get("metric_modes")
    if summary.get("semantic_threshold") is None:
        summary["semantic_threshold"] = manifest.get("semantic_threshold", b3.get("semantic_threshold"))
    if not summary.get("semantic_model"):
        summary["semantic_model"] = manifest.get("semantic_model", b3.get("semantic_model"))

    for key in req_summary:
        if key not in summary or summary.get(key) in (None, ""):
            msgs.append(f"summary missing field: {key}")
            ok = False

    expected_bases: Set[str] = set(corpus.get("base_ids") or [])
    scenarios = list(summary.get("scenarios") or [])
    if expected_bases and set(scenarios) != expected_bases:
        missing = expected_bases - set(scenarios)
        extra = set(scenarios) - expected_bases
        if missing:
            msgs.append(f"scenarios missing base_ids: {sorted(missing)}")
            ok = False
        if extra:
            msgs.append(f"scenarios extra base_ids: {sorted(extra)}")

    expected_profiles = list(b3.get("profiles") or [])
    profiles = list(summary.get("profiles") or [])
    if expected_profiles and profiles != expected_profiles:
        msgs.append(f"profiles expected {expected_profiles}, got {profiles}")

    if summary.get("eval_scenario_type") and summary.get("eval_scenario_type") != b3.get("eval_scenario_type"):
        msgs.append(
            f"eval_scenario_type: expected {b3.get('eval_scenario_type')!r}, "
            f"got {summary.get('eval_scenario_type')!r}"
        )

    exp_fetch = b3.get("fetch_k")
    if exp_fetch is not None and summary.get("fetch_k") not in (exp_fetch, *(b3.get("fetch_k_sensitivity_runs") or [])):
        msgs.append(
            f"fetch_k={summary.get('fetch_k')} not in protocol primary {exp_fetch} "
            f"nor sensitivity {b3.get('fetch_k_sensitivity_runs')}"
        )

    exp_modes = _as_modes(b3.get("metric_modes"))
    sum_modes = _as_modes(summary.get("metric_modes"))
    if exp_modes and sum_modes and sum_modes != exp_modes:
        msgs.append(f"summary metric_modes expected {exp_modes}, got {sum_modes}")

    results = summary.get("results") or {}
    cells_expected = len(scenarios) * len(profiles)
    cells_ok = 0
    semantic_seen = 0

    for sid in scenarios:
        for pname in profiles:
            cell = (results.get(sid) or {}).get(pname) or {}
            rc = cell.get("retrieval_returncode")
            if rc != 0:
                msgs.append(f"cell {sid}/{pname} retrieval_returncode={rc}")
                ok = False
                continue
            rpath = run_dir / sid / f"retrieval_{pname}.json"
            if not rpath.is_file():
                msgs.append(f"missing {rpath.relative_to(_ROOT)}")
                ok = False
                continue
            raw = json.loads(rpath.read_text(encoding="utf-8"))
            meta = raw.get("_meta") or {}
            msgs.extend(_check_meta(meta, req_meta, ctx=f"{sid}/{pname}"))
            if meta.get("max_rows", 0) not in (0, None):
                msgs.append(f"{sid}/{pname}: max_rows={meta.get('max_rows')} (expected 0 = full eval)")
                ok = False
            modes = _as_modes(meta.get("metric_modes"))
            if "semantic" in modes or any(k.startswith("semantic::") for k in raw if k != "_meta"):
                semantic_seen += 1
            elif exp_modes and "semantic" in exp_modes and not allow_missing_semantic:
                msgs.append(f"{sid}/{pname}: semantic not in metric_modes (got {modes})")
            cells_ok += 1

    msgs.insert(0, f"cells ok: {cells_ok}/{cells_expected}")
    if cells_ok != cells_expected:
        ok = False

    if exp_modes and "semantic" in exp_modes and semantic_seen == 0 and not allow_missing_semantic:
        msgs.append("no cell contains semantic metrics or semantic in metric_modes")
        ok = False
    elif allow_missing_semantic and semantic_seen == 0:
        msgs.append("WARN: semantic bundle not present (allow_missing_semantic=true)")

    chroma = summary.get("scenario_chroma") or {}
    exp_rows = int(corpus.get("expected_collection_rows") or 0)
    for sid in scenarios:
        row = chroma.get(sid) or {}
        verified = row.get("chroma_count_verified")
        if exp_rows and verified != exp_rows:
            msgs.append(f"{sid}: chroma_count_verified={verified} expected {exp_rows}")
            ok = False

    return ok, msgs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol",
        type=Path,
        default=_ROOT / "eval" / "config" / "final_research_protocol.yaml",
    )
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument(
        "--allow-missing-semantic",
        action="store_true",
        help="Accept runs before semantic bundle was enabled (canonical 20260513_114047).",
    )
    args = p.parse_args()

    protocol = _load_protocol(args.protocol.resolve())
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"run-dir not found: {run_dir}", file=sys.stderr)
        return 2

    ok, msgs = validate_b3_run_dir(
        run_dir,
        protocol,
        allow_missing_semantic=bool(args.allow_missing_semantic),
    )
    for line in msgs:
        print(line)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
