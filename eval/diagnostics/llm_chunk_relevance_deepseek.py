"""LLM-as-judge: DeepSeek scores chunk relevance (0–2) for trace slices.

Caches responses in JSONL so re-runs skip identical (prompt_version, question, chunk) pairs.
**Not for CI** without ``--dry-run`` (requires ``DEEPSEEK_API_KEY``).

Rubric (0 = irrelevant, 1 = related, 2 = directly supports answering the question).

Usage::

    python -m eval.diagnostics.llm_chunk_relevance_deepseek \\
        --trace eval/runs/<id>/trace.jsonl \\
        --out-jsonl eval/runs/<id>/llm_chunk_relevance.jsonl \\
        --max-queries 15 --top-chunks 2

Dry-run (no API)::

    python -m eval.diagnostics.llm_chunk_relevance_deepseek --dry-run --trace <trace> --out-jsonl <out>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _iter_query_traces(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("record") == "query_trace":
                yield o


def _chunk_texts(o: Dict[str, Any], top_k: int) -> List[str]:
    st = o.get("stage_01_dense_chroma_retrieval") or {}
    cands = st.get("candidates_ordered_by_chroma") or []
    out: List[str] = []
    for c in cands[: max(1, top_k)]:
        if isinstance(c, dict):
            out.append(str(c.get("text") or ""))
    return out


def _sha256_short(s: str, n: int = 16) -> str:
    h = hashlib.sha256((s or "").encode("utf-8")).hexdigest()
    return h[:n]


def parse_rubric_score(text: str) -> int:
    """Parse first 0/1/2 from model output."""
    t = (text or "").strip()
    for ch in t:
        if ch in "012":
            return int(ch)
    m = re.search(r"\b([0-2])\b", t)
    return int(m.group(1)) if m else 0


def _load_cache_keys(path: Path) -> Set[str]:
    keys: Set[str] = set()
    if not path.is_file():
        return keys
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = o.get("cache_key")
            if isinstance(k, str):
                keys.add(k)
    return keys


def _build_prompt(question: str, chunk: str, *, max_chunk_chars: int = 3500) -> str:
    ch = (chunk or "").strip()
    if len(ch) > max_chunk_chars:
        ch = ch[:max_chunk_chars] + "\n[truncated]"
    q = (question or "").strip()
    return (
        "You are an information retrieval judge. Score how relevant the passage is "
        "to answering the user question.\n\n"
        f"Question:\n{q}\n\nPassage:\n{ch}\n\n"
        "Reply with exactly one digit on the first line: 0, 1, or 2.\n"
        "0 = irrelevant or off-topic\n"
        "1 = tangentially related or partial context\n"
        "2 = directly useful to answer the question\n\n"
        "Score:"
    )


def run_llm_scores(
    *,
    trace_path: Path,
    out_jsonl: Path,
    prompt_version: str,
    max_queries: int,
    top_chunks: int,
    dry_run: bool,
    model: str,
    sleep_s: float,
) -> Dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cache_keys = _load_cache_keys(out_jsonl)
    fout = out_jsonl.open("a", encoding="utf-8")
    client = None
    if not dry_run:
        from core.graphrag.llm_provider import DeepSeekLLMClient

        try:
            client = DeepSeekLLMClient(model=model)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            print("Use --dry-run or set DEEPSEEK_API_KEY.", file=sys.stderr)
            raise SystemExit(2) from e

    n_written = 0
    n_cache_hit = 0
    n_queries = 0
    scores: List[int] = []

    try:
        for o in _iter_query_traces(trace_path):
            if max_queries > 0 and n_queries >= max_queries:
                break
            idx = int(o.get("eval_row_index", -1))
            q = str(o.get("question_text") or "").strip()
            if not q:
                continue
            chunks = _chunk_texts(o, top_chunks)
            if not any(chunks):
                continue
            n_queries += 1

            for rank, chunk in enumerate(chunks, start=1):
                if not chunk.strip():
                    continue
                qsh = _sha256_short(q, 24)
                csh = _sha256_short(chunk, 24)
                cache_key = f"{prompt_version}|{qsh}|{csh}|{rank}"
                if cache_key in cache_keys:
                    n_cache_hit += 1
                    continue

                raw = ""
                score = -1
                if dry_run:
                    score = -1
                else:
                    assert client is not None
                    prompt = _build_prompt(q, chunk)
                    raw = client.complete(prompt, temperature=0.0, max_tokens=8)
                    score = parse_rubric_score(raw)
                    if sleep_s > 0:
                        time.sleep(sleep_s)

                row = {
                    "record": "llm_chunk_relevance",
                    "eval_row_index": idx,
                    "chunk_rank": rank,
                    "prompt_version": prompt_version,
                    "model": model,
                    "cache_key": cache_key,
                    "score": score,
                    "dry_run": bool(dry_run),
                    "question_sha256": qsh,
                    "chunk_sha256": csh,
                    "raw_head": (raw[:200] if raw else ""),
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                cache_keys.add(cache_key)
                n_written += 1
                if score >= 0:
                    scores.append(int(score))
    finally:
        fout.close()

    mean = sum(scores) / max(1, len(scores)) if scores else None
    return {
        "schema": "llm_chunk_relevance_summary_v1",
        "trace": str(trace_path.as_posix()),
        "out_jsonl": str(out_jsonl.as_posix()),
        "prompt_version": prompt_version,
        "max_queries": max_queries,
        "top_chunks": top_chunks,
        "dry_run": dry_run,
        "queries_processed": n_queries,
        "rows_appended": n_written,
        "cache_hits_skipped": n_cache_hit,
        "mean_score_non_dry": mean,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out-jsonl", type=Path, required=True)
    p.add_argument("--prompt-version", default="v1")
    p.add_argument("--max-queries", type=int, default=20)
    p.add_argument("--top-chunks", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--model", default=os.getenv("LLM_MODEL", "deepseek-chat"))
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds between API calls.")
    args = p.parse_args()

    if args.dry_run:
        summary = run_llm_scores(
            trace_path=args.trace.resolve(),
            out_jsonl=args.out_jsonl.resolve(),
            prompt_version=str(args.prompt_version),
            max_queries=min(3, max(1, int(args.max_queries))),
            top_chunks=min(2, max(1, int(args.top_chunks))),
            dry_run=True,
            model=str(args.model),
            sleep_s=0.0,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    summary = run_llm_scores(
        trace_path=args.trace.resolve(),
        out_jsonl=args.out_jsonl.resolve(),
        prompt_version=str(args.prompt_version),
        max_queries=int(args.max_queries),
        top_chunks=max(1, int(args.top_chunks)),
        dry_run=False,
        model=str(args.model),
        sleep_s=float(args.sleep),
    )
    meta_path = args.out_jsonl.resolve().with_suffix(".summary.json")
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Wrote", meta_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
