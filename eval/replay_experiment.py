"""
Replay a named experiment from eval/experiments_registry.yaml (single source of truth).

  python -m eval.replay_experiment --registry eval/experiments_registry.yaml list
  python -m eval.replay_experiment run <experiment_id> [--dry-run]
  python -m eval.replay_experiment run <id> -e CHROMA_COLLECTION=my_col

Merges: os.environ (parent) <- defaults[*inherit] <- experiment.env (with ${VAR:-def} expansion).
Optional key rag_profile: applies apply_profile() before env merge.
Command: first token "python" is replaced with sys.executable.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.rag_profile_env import apply_profile  # noqa: E402

_VAR_DEFAULT = re.compile(r"\$\{([^}:]+):-([^}]*)\}")


def _expand(s: str, extra: Optional[Mapping[str, str]] = None) -> str:
    extra = extra or {}

    def repl(m: re.Match[str]) -> str:
        var, dfl = m.group(1), m.group(2)
        v = os.environ.get(var)
        if v is None or v == "":
            v = extra.get(var)
        if v is None or v == "":
            return dfl
        return str(v)

    return _VAR_DEFAULT.sub(repl, s)


def _expand_mapping(m: Mapping[str, Any], extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in m.items():
        if v is None:
            continue
        out[str(k)] = _expand(str(v), extra)
    return out


def _default_registry_path() -> Path:
    p = os.getenv("EXPERIMENTS_REGISTRY", "").strip()
    if p:
        return Path(p)
    return _ROOT / "eval" / "experiments_registry.yaml"


def _load_registry(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("registry root must be a mapping")
    return data


def _merge_inherit(defaults: Mapping[str, Any], inherit: List[str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    for name in inherit:
        block = defaults.get(name)
        if not isinstance(block, dict):
            raise KeyError(f"Unknown defaults block: {name!r}")
        merged.update(_expand_mapping(block))
    return merged


def _resolve_cmd(cmd: List[str]) -> List[str]:
    if not cmd:
        raise ValueError("empty command")
    out = list(cmd)
    if out[0] == "python":
        out[0] = sys.executable
    return out


def cmd_list(registry: Mapping[str, Any], exp: Mapping[str, Any]) -> List[str]:
    raw = exp.get("command")
    if not isinstance(raw, list) or not raw:
        raise ValueError("experiment.command must be a non-empty list")
    return [_expand(str(x)) for x in raw]


def build_child_env(
    registry: Mapping[str, Any],
    exp: Mapping[str, Any],
    *,
    extra: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    defaults = registry.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("registry.defaults must be a mapping")

    inherit = exp.get("inherit") or []
    if not isinstance(inherit, list):
        raise ValueError("experiment.inherit must be a list")

    merged = _merge_inherit(defaults, [str(x) for x in inherit])
    env_block = exp.get("env") or {}
    if not isinstance(env_block, dict):
        raise ValueError("experiment.env must be a mapping")
    merged.update(_expand_mapping(env_block, extra))
    return merged


def cmd_list_public(registry: Mapping[str, Any], exp: Mapping[str, Any]) -> List[str]:
    return _resolve_cmd(cmd_list(registry, exp))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(description="List or replay experiments from experiments_registry.yaml")
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="YAML registry path (default: eval/experiments_registry.yaml or EXPERIMENTS_REGISTRY).",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("list", help="Print experiment ids and summaries")

    p_run = sub.add_parser("run", help="Run one experiment")
    p_run.add_argument("experiment_id", help="Key under experiments:")
    p_run.add_argument("--dry-run", action="store_true", help="Print merged env additions and command only")
    p_run.add_argument(
        "-e",
        "--env",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Extra KEY=VAL for this run only (merged last).",
    )

    args = parser.parse_args()
    path = args.registry or _default_registry_path()
    if not path.is_file():
        print("Registry not found:", path, file=sys.stderr)
        return 2

    data = _load_registry(path)
    experiments = data.get("experiments") or {}
    if not isinstance(experiments, dict):
        print("registry.experiments must be a mapping", file=sys.stderr)
        return 2

    if args.action == "list":
        for eid in sorted(experiments.keys()):
            exp = experiments[eid]
            if not isinstance(exp, dict):
                continue
            st = exp.get("status", "?")
            summ = exp.get("summary", "")
            print(f"{eid}\t[{st}]\t{summ}")
        return 0

    if args.action == "run":
        eid = args.experiment_id
        exp = experiments.get(eid)
        if not isinstance(exp, dict):
            print("Unknown experiment:", eid, file=sys.stderr)
            return 2

        extra_kv: Dict[str, str] = {}
        for item in getattr(args, "env", []) or []:
            if "=" not in item:
                print("Bad --env, expected KEY=VAL:", item, file=sys.stderr)
                return 2
            k, v = item.split("=", 1)
            extra_kv[k.strip()] = v

        rag_profile = exp.get("rag_profile")
        if rag_profile:
            apply_profile(str(rag_profile))

        child_env_updates = build_child_env(data, exp, extra=extra_kv)
        child_env_updates.update(extra_kv)

        cmd = cmd_list_public(data, exp)

        if args.dry_run:
            print("Command:", cmd)
            print("Env overrides:")
            for k in sorted(child_env_updates.keys()):
                print(f"  {k}={child_env_updates[k]}")
            return 0

        full_env = dict(os.environ)
        full_env.update(child_env_updates)
        print("[replay_experiment]", eid, "->", " ".join(cmd), flush=True)
        proc = subprocess.run(cmd, cwd=str(_ROOT), env=full_env)
        return int(proc.returncode)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
