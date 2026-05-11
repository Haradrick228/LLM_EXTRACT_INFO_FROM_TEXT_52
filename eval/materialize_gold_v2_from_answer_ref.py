"""
Материализация gold v2 из eval/eval_set_with_answer_ref.jsonl — без LLM.

Каждая строка eval_set: document чанка = answer_ref + недостающие references (чтобы
каждая цитата была подстрокой document для метрик). Поле reference_quote = references[0].
В eval_set_gold_v2.jsonl пишутся только question, reference_quote, references, answer_ref
(без chunk_id и source; соответствие строке fixture — по порядку, см. metadata.row_index).

Запуск (перезапись eval_set_gold_v2 + fixtures):
  python -m eval.materialize_gold_v2_from_answer_ref --n 100 --yes
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
SRC = _ROOT / "eval" / "eval_set_with_answer_ref.jsonl"
FIX = _ROOT / "eval" / "fixtures" / "chunks_sample.jsonl"
GOLD = _ROOT / "eval" / "eval_set_gold_v2.jsonl"
META = _ROOT / "eval" / "eval_set_gold_v2.meta.json"


def _chunk_document(r: dict) -> str:
    doc = (r.get("answer_ref") or "").strip()
    for ref in r.get("references") or []:
        ref = (ref or "").strip()
        if ref and ref not in doc:
            doc = doc + "\n\n" + ref
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="Сколько первых строк взять (макс. размер исходника).")
    ap.add_argument("--yes", action="store_true", help="Разрешить перезапись gold/fixture/meta.")
    args = ap.parse_args()
    if not args.yes:
        print("Нужен флаг --yes для перезаписи файлов.", flush=True)
        return 2

    rows = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    n = min(args.n, len(rows))
    rows = rows[:n]

    fix_lines = []
    gold_lines = []
    for i, r in enumerate(rows):
        cid = f"corpus_answer_ref::{i:04d}"
        doc = _chunk_document(r)
        refs = r.get("references") or []
        rq = (refs[0] if refs else "").strip()
        if not rq:
            print(f"skip row {i}: empty references", flush=True)
            continue
        if rq not in doc:
            print(f"skip row {i}: quote not in doc", flush=True)
            continue

        fix_lines.append(
            json.dumps(
                {"id": cid, "document": doc, "metadata": {"row_index": i}},
                ensure_ascii=False,
            )
        )
        gold_obj = {
            "question": (r.get("question") or "").strip(),
            "reference_quote": rq,
            "references": list(refs),
            "answer_ref": (r.get("answer_ref") or "").strip(),
        }
        gold_lines.append(json.dumps(gold_obj, ensure_ascii=False))

    FIX.parent.mkdir(parents=True, exist_ok=True)
    FIX.write_text("\n".join(fix_lines) + "\n", encoding="utf-8")
    GOLD.write_text("\n".join(gold_lines) + "\n", encoding="utf-8")

    meta = {
        "schema": "gold_v2",
        "version": 4,
        "line_count": len(gold_lines),
        "corpus_type": "corpus_migrated",
        "corpus_type_note": "Строки перенесены из eval/eval_set_with_answer_ref.jsonl (реальные вопросы и цитаты корпуса). LLM не использовался. Чанк = объединение answer_ref и references для substring-метрик. Gold-строка без chunk_id/source; соответствие fixture — по порядку строк (индекс = row_index в metadata чанка).",
        "source_eval_file": "eval/eval_set_with_answer_ref.jsonl",
        "materializer": "eval/materialize_gold_v2_from_answer_ref.py",
        "built_unix": int(time.time()),
        "fixture_chunks": "eval/fixtures/chunks_sample.jsonl",
    }
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(gold_lines)} -> {GOLD.relative_to(_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
