"""Проверка, что все коллекции из embedding_bases.yaml существуют и имеют ожидаемый count (как число строк фикстуры)."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.chroma_utils import chroma_http_client  # noqa: E402
from eval.indexing.chroma_embedding_function_factory import build_chroma_embedding_function  # noqa: E402
from eval.datasets.load_gold_fixture_chunks import load_fixture_batches  # noqa: E402
from eval.experiments.phase_b_collections_registry import load_registry, list_bases  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, default=None)
    args = p.parse_args()

    data = load_registry(args.registry)
    fix = _ROOT / str(data.get("defaults", {}).get("fixture_rel", "eval/fixtures/chunks_sample.jsonl"))
    ids, _, _ = load_fixture_batches(fix)
    expected = len(ids)

    client = chroma_http_client()
    ok = True
    print("base_id\tcollection\tcount\texpected")
    for b in list_bases(data):
        merged = dict(data.get("defaults") or {})
        merged.update(b)
        name = str(merged["collection"])
        ef = build_chroma_embedding_function(merged)
        col = client.get_collection(name, embedding_function=ef)
        n = col.count()
        row_ok = n == expected
        ok = ok and row_ok
        mark = "OK" if row_ok else "MISMATCH"
        print(f"{b['base_id']}\t{name}\t{n}\t{expected}\t{mark}")

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
