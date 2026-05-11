"""Run a subprocess with env vars from eval/rag_profiles.yaml applied first."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.rag_profile_env import apply_profile  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply RAG profile then exec a command (e.g. uvicorn).",
        usage="python -m eval.run_with_profile --profile NAME COMMAND [ARGS...]",
    )
    parser.add_argument("--profile", required=True, help="Profile name from eval/rag_profiles.yaml")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command and arguments (prefix with -- if first arg is -)")
    args = parser.parse_args()
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("Error: missing command after --profile, e.g. uvicorn ml_service.main:app --host 0.0.0.0 --port 8000", file=sys.stderr)
        sys.exit(2)

    apply_profile(args.profile)
    os.environ["RAG_PROFILE"] = args.profile
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
