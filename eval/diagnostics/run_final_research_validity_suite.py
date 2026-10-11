"""Run metric-validity checks from ``final_research_protocol.yaml`` (offline + optional Chroma)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_protocol(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_metric(json_path: Path, key: str) -> Optional[float]:
    if not json_path.is_file():
        return None
    data = json.loads(json_path.read_text(encoding="utf-8"))
    v = data.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _chroma_ok(host: str, port: int) -> bool:
    try:
        from eval.chroma_utils import chroma_http_client

        chroma_http_client(host=host, port=port).heartbeat()
        return True
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--protocol",
        type=Path,
        default=_ROOT / "eval" / "config" / "final_research_protocol.yaml",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=_ROOT / "eval" / "runs" / "final_research_validity_suite.json",
    )
    p.add_argument("--skip-chroma", action="store_true", help="Do not run neg-control retrieval even if Chroma is up.")
    args = p.parse_args()

    protocol = _load_protocol(args.protocol.resolve())
    canon = protocol.get("canonical_artifacts") or {}
    criteria = protocol.get("validity_criteria") or {}
    b3 = protocol.get("retrieval_matrix_b3") or {}
    anchor = protocol.get("anchor_dense_trace") or {}

    report: Dict[str, Any] = {
        "protocol_version": protocol.get("protocol_version"),
        "checks": {},
    }
    all_pass = True

    # --- External sanity ---
    sanity_dir = _ROOT / str((canon.get("external_sanity") or {}).get("run_dir", ""))
    sanity_ret = sanity_dir / "retrieval.json"
    s_key = str((criteria.get("external_sanity") or {}).get("metric", "substring::recall@10"))
    s_min = float((criteria.get("external_sanity") or {}).get("min_value", 0.5))
    s_val = _read_metric(sanity_ret, s_key) or _read_metric(sanity_ret, "recall@10")
    s_ok = s_val is not None and s_val >= s_min
    report["checks"]["external_sanity"] = {
        "path": str(sanity_ret.relative_to(_ROOT)) if sanity_ret.is_file() else str(sanity_ret),
        "metric": s_key,
        "value": s_val,
        "min_value": s_min,
        "pass": s_ok,
    }
    all_pass = all_pass and s_ok

    # --- Shuffled eval materialization ---
    neg = canon.get("neg_control") or {}
    shuffled = _ROOT / str(neg.get("shuffled_eval", "eval/runs/_neg_control/shuffled_questions_gold_v2.jsonl"))
    shuffled.parent.mkdir(parents=True, exist_ok=True)
    if not shuffled.is_file():
        rc = subprocess.call(
            [
                sys.executable,
                "-m",
                "eval.diagnostics.shuffle_eval_questions_neg_control",
                "--eval-jsonl",
                str(_ROOT / protocol["eval_set"]["path"]),
                "--out-jsonl",
                str(shuffled),
            ],
            cwd=str(_ROOT),
        )
        report["checks"]["neg_control_shuffle"] = {"created": rc == 0, "path": str(shuffled.relative_to(_ROOT))}
        if rc != 0:
            all_pass = False
    else:
        report["checks"]["neg_control_shuffle"] = {"created": True, "path": str(shuffled.relative_to(_ROOT)), "reused": True}

    # --- Neg-control retrieval (needs Chroma) ---
    host = str(b3.get("chroma_host", "127.0.0.1"))
    port = int(b3.get("chroma_port", 18000))
    neg_cfg = neg if isinstance(neg, dict) else {}
    neg_out = _ROOT / str(neg_cfg.get("retrieval_out", shuffled.parent / "retrieval_neg_control.json"))
    neg_metric = str((criteria.get("neg_control") or {}).get("metric", "substring::recall@10"))
    b3_canon = canon.get("b3_matrix_canonical") or canon.get("b3_matrix_6x3_fetch100") or {}
    honest_dir = _ROOT / str(b3_canon.get("run_dir", ""))
    honest_spec = neg_cfg.get("honest_cell") or {}
    honest_sid = str(honest_spec.get("scenario", "full_frida"))
    honest_prof = str(honest_spec.get("profile", "rerank_jina_v3"))
    honest_cell = honest_dir / honest_sid / f"retrieval_{honest_prof}.json"
    honest_val = _read_metric(honest_cell, neg_metric) or _read_metric(honest_cell, "recall@10")

    neg_val: Optional[float] = _read_metric(neg_out, neg_metric) or _read_metric(neg_out, "recall@10")
    chroma_up = _chroma_ok(host, port) and not args.skip_chroma

    def _build_neg_control_cmd() -> Optional[List[str]]:
        if not honest_cell.is_file():
            return None
        meta = json.loads(honest_cell.read_text(encoding="utf-8")).get("_meta") or {}
        embed = str(meta.get("embed_model") or "")
        coll = str(meta.get("chroma_collection") or "")
        if not embed or not coll:
            return None
        fk = int(meta.get("fetch_k") or b3.get("fetch_k") or 40)
        modes = meta.get("metric_modes")
        if isinstance(modes, list):
            modes_s = ",".join(str(m) for m in modes)
        else:
            modes_s = str(b3.get("metric_modes", "substring,token_f1,semantic"))
        cmd = [
            sys.executable,
            "-m",
            "eval.retrieval.run_retrieval_eval",
            "--eval-file",
            str(shuffled),
            "--embed-model",
            embed,
            "--fetch-k",
            str(fk),
            "--k",
            str(b3.get("k", "5,10,20")),
            "--metric-modes",
            modes_s,
            "--semantic-threshold",
            str(meta.get("semantic_threshold", b3.get("semantic_threshold", 0.62))),
            "--semantic-model",
            str(meta.get("semantic_model", b3.get("semantic_model", ""))),
            "--chroma-collection",
            coll,
            "--chroma-host",
            host,
            "--chroma-port",
            str(port),
            "--out-json",
            str(neg_out),
        ]
        rerank_model = str(meta.get("rerank_model") or "").strip()
        if rerank_model and rerank_model != "(none)":
            cmd.extend(
                [
                    "--rerank-model",
                    rerank_model,
                    "--rerank-mode",
                    str(meta.get("rerank_mode", "cross")),
                ]
            )
        return cmd

    if chroma_up:
        neg_out.parent.mkdir(parents=True, exist_ok=True)
        cmd = _build_neg_control_cmd()
        if cmd is not None:
            subprocess.call(cmd, cwd=str(_ROOT))
            neg_val = _read_metric(neg_out, neg_metric) or _read_metric(neg_out, "recall@10")
        else:
            report["checks"]["neg_control_retrieval"] = {
                "pass": False,
                "note": f"missing honest cell meta: {honest_cell}",
            }
            all_pass = False

    neg_ok = False
    if honest_val is not None and neg_val is not None:
        gap = honest_val - neg_val
        ratio = (honest_val / neg_val) if neg_val > 1e-9 else float("inf")
        neg_ok = gap >= 0.03 or ratio >= 2.0
    if "neg_control_retrieval" not in report:
        neg_entry: Dict[str, Any] = {
            "honest_path": str(honest_cell.relative_to(_ROOT)) if honest_cell.is_file() else str(honest_cell),
            "honest_scenario": honest_sid,
            "honest_profile": honest_prof,
            "honest_value": honest_val,
            "shuffled_path": str(neg_out.relative_to(_ROOT)) if neg_out.is_file() else str(neg_out),
            "shuffled_value": neg_val,
            "metric": neg_metric,
            "chroma_available": chroma_up,
            "pass": neg_ok if neg_val is not None else None,
            "skipped": neg_val is None and not chroma_up,
        }
        if honest_val is not None and neg_val is not None:
            neg_entry["gap"] = honest_val - neg_val
            neg_entry["ratio"] = (honest_val / neg_val) if neg_val > 1e-9 else None
        report["checks"]["neg_control_retrieval"] = neg_entry
        if neg_val is None:
            report["checks"]["neg_control_retrieval"]["note"] = (
                "Start Chroma (docker compose) and re-run suite for neg-control retrieval."
            )
            report["neg_control_pending"] = True
        else:
            all_pass = all_pass and neg_ok
            report["neg_control_pending"] = False

    # --- Ladder A–D + Wilson on anchor trace ---
    anchor_run = _ROOT / str(anchor.get("canonical_run_dir", ""))
    trace = anchor_run / "trace.jsonl"
    ladder_out = anchor_run / "metrics_ladder_full_corpus.json"
    corpus_jsonl = _ROOT / str(protocol["corpus"]["corpus_export_jsonl"])
    ladder_ok = False
    if trace.is_file():
        ladder_argv = [
            "--trace",
            str(trace),
            "--gold-jsonl",
            str(_ROOT / protocol["eval_set"]["path"]),
            "--hit-fn",
            str(criteria.get("ladder_hit_fn", "substring")),
            "--k-primary",
            str(int(criteria.get("ladder_primary_k", 10))),
            "--out-json",
            str(ladder_out),
        ]
        if corpus_jsonl.is_file():
            ladder_argv.extend(["--corpus-jsonl", str(corpus_jsonl)])
        rc = subprocess.call(
            [sys.executable, "-m", "eval.diagnostics.metrics_ladder_funnel", *ladder_argv],
            cwd=str(_ROOT),
        )
        ladder_ok = rc == 0 and ladder_out.is_file()
        if ladder_ok:
            ladder_data = json.loads(ladder_out.read_text(encoding="utf-8"))
            frac = ladder_data.get("class_fractions") or {}
            report["checks"]["metrics_ladder"] = {
                "path": str(ladder_out.relative_to(_ROOT)),
                "class_fraction_A": frac.get("A"),
                "class_fraction_D": frac.get("D"),
                "conditional_given_oracle_pool": ladder_data.get("conditional_given_oracle_pool"),
                "wilson_hit_at_k_post": ladder_data.get("wilson_hit_at_k_post"),
                "pass": True,
            }
        else:
            report["checks"]["metrics_ladder"] = {"pass": False, "rc": rc}
            all_pass = False
    else:
        report["checks"]["metrics_ladder"] = {"pass": False, "note": f"missing trace: {trace}"}
        all_pass = False

    # --- Semantic probe on same trace ---
    probe_out = anchor_run / "semantic_threshold_probe.json"
    if trace.is_file():
        prc = subprocess.call(
            [
                sys.executable,
                "-m",
                "eval.diagnostics.semantic_threshold_probe",
                "--trace",
                str(trace),
                "--gold-jsonl",
                str(_ROOT / protocol["eval_set"]["path"]),
                "--out-json",
                str(probe_out),
                "--max-rows",
                "30",
            ],
            cwd=str(_ROOT),
        )
        report["checks"]["semantic_threshold_probe"] = {
            "path": str(probe_out.relative_to(_ROOT)),
            "pass": prc == 0 and probe_out.is_file(),
        }
        all_pass = all_pass and report["checks"]["semantic_threshold_probe"]["pass"]

    report["overall_pass"] = all_pass
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("overall_pass=", all_pass)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
