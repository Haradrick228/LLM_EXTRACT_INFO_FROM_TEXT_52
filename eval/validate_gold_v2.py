"""
Проверки целостности gold v2 ↔ fixtures для метрик (substring по references).

  python -m eval.validate_gold_v2

Режимы (авто по meta / полям):
- synthetic: answer_ref == document чанка (как generate_gold_v2_corpus).
- chroma_curated / corpus_migrated / наличие reference_quote: answer_ref — свободная формулировка;
  каждая строка из references и reference_quote (если есть) — подстрока document.
Сопоставление gold ↔ fixture: по индексу строки (i-я строка gold = i-й чанк во fixture).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = _ROOT / "eval" / "fixtures" / "chunks_sample.jsonl"
GOLD = _ROOT / "eval" / "eval_set_gold_v2.jsonl"
META = _ROOT / "eval" / "eval_set_gold_v2.meta.json"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def validate() -> Tuple[List[str], dict]:
    errors: List[str] = []
    chunks = load_jsonl(FIXTURE)
    gold_rows = load_jsonl(GOLD)
    meta = json.loads(META.read_text(encoding="utf-8")) if META.is_file() else {}

    fix_ids = [c.get("id") for c in chunks]
    summary = {
        "fixture_rows": len(chunks),
        "gold_rows": len(gold_rows),
        "unique_fixture_ids": len(set(fix_ids)),
        "corpus_type": meta.get("corpus_type", "unknown"),
        "alignment": "row_index",
    }

    relaxed = meta.get("corpus_type") in ("chroma_curated", "corpus_migrated") or any(
        "reference_quote" in r for r in gold_rows
    )

    if len(chunks) != len(gold_rows):
        errors.append(f"Разное число строк: fixture={len(chunks)} gold={len(gold_rows)}")

    if len(fix_ids) != len(set(fix_ids)):
        errors.append("Дубликаты id во fixture")

    lc = meta.get("line_count")
    if lc is not None and int(lc) != len(gold_rows):
        errors.append(f"meta line_count={lc} не совпадает с числом строк gold={len(gold_rows)}")

    for i, r in enumerate(gold_rows):
        prefix = f"gold[{i}]"
        if i >= len(chunks):
            errors.append(f"{prefix}: нет парной строки во fixture (индекс за пределами)")
            continue
        ch = chunks[i]
        doc = ch["document"]
        if not r.get("question", "").strip():
            errors.append(f"{prefix}: пустой question")
        refs = r.get("references")
        if not isinstance(refs, list) or not refs:
            errors.append(f"{prefix}: references должен быть непустым list")
            continue

        rq = (r.get("reference_quote") or "").strip()
        if rq:
            if rq not in doc:
                errors.append(f"{prefix}: reference_quote не подстрока document")
            if refs[0] != rq and rq not in refs:
                errors.append(f"{prefix}: reference_quote должен совпадать с references[0] или входить в references")

        for j, ref in enumerate(refs):
            if not isinstance(ref, str) or not ref.strip():
                errors.append(f"{prefix}: references[{j}] пустая строка")
            elif ref not in doc:
                errors.append(f"{prefix}: references[{j}] не подстрока document")

        ar = (r.get("answer_ref") or "").strip()
        if relaxed:
            if len(ar) < 12:
                errors.append(f"{prefix}: answer_ref слишком короткий для chroma_curated")
        else:
            if ar != doc:
                errors.append(f"{prefix}: answer_ref должен совпадать с document (synthetic)")

    return errors, summary


def main() -> int:
    errs, summary = validate()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errs:
        for e in errs:
            print("ERROR:", e, file=sys.stderr)
        return 1
    print("OK: gold v2 согласован с fixture; references (и цитата) — подстроки чанка.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
