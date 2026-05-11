"""
Синтетический корпус (без Chroma) — в основном для автотестов и CI.

Для рабочего golden-датасета используйте реальные чанки + LLM:
  python -m eval.build_gold_v2_from_chroma --collection default --n 50 --yes

  python eval/generate_gold_v2_corpus.py --n 100
  python eval/generate_gold_v2_corpus.py --n 50

Цитата в `references[0]` — дословный подотрезок `document`; answer_ref = document.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = _ROOT / "eval" / "fixtures" / "chunks_sample.jsonl"
GOLD_PATH = _ROOT / "eval" / "eval_set_gold_v2.jsonl"
META_PATH = _ROOT / "eval" / "eval_set_gold_v2.meta.json"

# Разнообразные формулировки тем (циклически); без внешних цитат — только для генерации.
_THEMES = [
    "цели урока и критерии успеха",
    "самооценка и рефлексия учащихся",
    "дифференциация заданий по уровню сложности",
    "обратная связь и корректировка темпа",
    "проектная деятельность в группе",
    "работа с источниками и проверка фактов",
    "метапредметные результаты",
    "культура ошибки и безопасная среда",
    "связь с локальным контекстом региона",
    "оценивание по рубрикам",
    "партнёрское обучение",
    "инклюзия и доступность материалов",
    "цифровая гигиена и этика поиска",
    "планирование долгосрочной траектории",
    "связь теории с практикой вне школы",
    "мотивация и автономия ученика",
    "языковая поддержка билингвов",
    "межпредметные связи",
    "безопасность при лабораторных работах",
    "подготовка к итоговой аттестации",
]


def _chunk_and_gold(i: int) -> tuple[dict, dict]:
    """i от 0 до n-1; EDU-номера с 1."""
    idx = i + 1
    theme = _THEMES[i % len(_THEMES)]
    quote = (
        f"Опорная формулировка блока {idx}: материал связывает тему «{theme}» "
        f"с измеримыми действиями на занятии."
    )
    doc = (
        f"Блок EDU-{idx:03d} раскрывает тему «{theme}». {quote} "
        f"Преподаватель фиксирует прогресс, использует рубрики и при необходимости "
        f"корректирует задания после короткого опроса группы (шаг {idx % 5 + 1} из цикла)."
    )
    src = f"syn_edu_{idx:03d}"
    cid = f"{src}::0"
    chunk = {
        "id": cid,
        "document": doc,
        "metadata": {"source": src, "chunk_index": 0, "edu_index": idx},
    }
    gold = {
        "question": f"Как сформулирована опорная мысль блока EDU-{idx:03d} и какая тема у блока?",
        "reference_quote": quote,
        "references": [quote],
        "answer_ref": doc,
    }
    return chunk, gold


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="Число чанков = число вопросов (50–100 рекомендуется).")
    args = p.parse_args()
    if args.n < 10:
        raise SystemExit("--n must be >= 10")

    chunks: list[dict] = []
    golds: list[dict] = []
    for i in range(args.n):
        c, g = _chunk_and_gold(i)
        chunks.append(c)
        golds.append(g)

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
        encoding="utf-8",
    )
    GOLD_PATH.write_text(
        "\n".join(json.dumps(g, ensure_ascii=False) for g in golds) + "\n",
        encoding="utf-8",
    )

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    meta["line_count"] = args.n
    meta["corpus_type"] = "synthetic"
    meta["corpus_type_note"] = "Сгенерировано generate_gold_v2_corpus.py; не использовать как продакшн-gold."
    meta["generator"] = "eval/generate_gold_v2_corpus.py"
    meta["generator_note"] = (
        f"Синтетический корпус из {args.n} чанков; references — подстроки document; answer_ref=document."
    )
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.n} chunks -> {FIXTURE_PATH.relative_to(_ROOT)}")
    print(f"Wrote {args.n} gold rows -> {GOLD_PATH.relative_to(_ROOT)}")
    print(f"Updated {META_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
