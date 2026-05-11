"""
Строит реальный golden v2 из чанков Chroma: вопрос + ответ + дословная цитата (подстрока чанка).

Использует локальную Ollama (LLM_PROVIDER=ollama по умолчанию для этого скрипта).
Пишет пару файлов, согласованных для eval/validate_gold_v2:
  - eval/fixtures/chunks_sample.jsonl — только выбранные чанки
  - eval/eval_set_gold_v2.jsonl — по одной строке на чанк

Пример:
  set LLM_PROVIDER=ollama
  python -m eval.build_gold_v2_from_chroma --collection default --n 50 --yes

Фаза B не требуется: только выборка из существующей коллекции.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.chroma_utils import chroma_http_client, get_collection  # noqa: E402


def _embed_model() -> str:
    local = _ROOT / "DATA" / "hf_embedders" / "sentence-transformers__paraphrase-multilingual-mpnet-base-v2"
    if (local / "config.json").is_file():
        return str(local)
    return os.getenv(
        "CHROMA_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", t, re.I | re.DOTALL)
    candidates = []
    if fence:
        candidates.append(fence.group(1))
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        candidates.append(t[s : e + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _llm_row(doc: str, llm) -> Tuple[Optional[Dict[str, str]], str]:
    """Возвращает (payload|None, raw_text). payload: question, answer_ref, reference_quote."""
    prompt = (
        "Ниже один фрагмент текста из базы знаний (чанк). Придумай ВОПРОС на русском, на который можно "
        "ответить ТОЛЬКО по этому фрагменту (как в RAG: ответ должен опираться на текст).\n"
        "Запиши также КРАТКИЙ ответ на русском (answer_ref) — своими словами, но без выдуманных фактов "
        "вне фрагмента.\n"
        "Обязательно укажи reference_quote: ПРЯМАЯ ЦИТАТА из фрагмента — скопируй дословно непрерывный "
        "кусок из текста (от 40 до 500 символов), который лучше всего подтверждает ответ.\n\n"
        "Верни ТОЛЬКО один JSON-объект без пояснений, вида:\n"
        '{"question":"...","answer_ref":"...","reference_quote":"..."}\n\n'
        "ФРАГМЕНТ:\n<<<CHUNK>>>\n"
        f"{doc}\n"
        "<<<END>>>"
    )
    raw = llm.complete(prompt, temperature=0.15, max_tokens=900)
    data = _extract_json_object(raw)
    if not isinstance(data, dict):
        return None, raw
    q = (data.get("question") or "").strip()
    a = (data.get("answer_ref") or "").strip()
    rq = (data.get("reference_quote") or "").strip()
    if not q or not a or not rq:
        return None, raw
    return {"question": q, "answer_ref": a, "reference_quote": rq}, raw


def _sample_chunks(col, n: int, seed: int) -> List[Dict[str, Any]]:
    total = col.count()
    if total == 0:
        return []
    n = min(n, total)
    random.seed(seed)
    pool = min(8000, total)
    start = random.randint(0, max(0, total - pool))
    batch = col.get(
        include=["documents", "metadatas"],
        limit=pool,
        offset=start,
    )
    ids = batch.get("ids") or []
    docs = batch.get("documents") or []
    metas = batch.get("metadatas") or []
    idxs = list(range(len(ids)))
    random.shuffle(idxs)
    out: List[Dict[str, Any]] = []
    for i in idxs[:n]:
        doc = docs[i] if i < len(docs) else ""
        if not (doc or "").strip():
            continue
        out.append(
            {
                "id": ids[i],
                "document": doc,
                "metadata": metas[i] if i < len(metas) else {},
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "default"))
    parser.add_argument("--n", type=int, default=50, help="Сколько чанков (и строк gold).")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Разрешить перезапись eval/eval_set_gold_v2.jsonl и eval/fixtures/chunks_sample.jsonl",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Повторов LLM на чанк при невалидной цитате.",
    )
    args = parser.parse_args()

    if not args.yes:
        print("Укажите --yes для перезаписи eval/eval_set_gold_v2.jsonl и eval/fixtures/chunks_sample.jsonl", flush=True)
        return 2

    os.environ.setdefault("RAG_EVAL_CHROMA_HOST", "127.0.0.1")
    os.environ.setdefault("RAG_EVAL_CHROMA_PORT", "18000")
    os.environ.setdefault("LLM_PROVIDER", "ollama")

    embed = _embed_model()
    client = chroma_http_client()
    col = get_collection(client, args.collection, embed)

    picked = _sample_chunks(col, args.n, args.seed)
    if len(picked) < min(5, args.n):
        print(f"Слишком мало чанков с текстом (получено {len(picked)})", flush=True)
        return 3

    from core.graphrag.llm_provider import build_llm_client  # noqa: E402

    llm = build_llm_client()

    gold_lines: List[str] = []
    fix_lines: List[str] = []
    errors = 0

    for i, row in enumerate(picked):
        cid = row["id"]
        doc = row["document"]
        meta = row.get("metadata") or {}

        ok_payload = None
        last_raw = ""
        for attempt in range(args.max_retries + 1):
            payload, last_raw = _llm_row(doc, llm)
            if not payload:
                continue
            rq = payload["reference_quote"]
            if rq not in doc:
                continue
            if len(rq) < 25:
                continue
            ok_payload = payload
            break

        if not ok_payload:
            print(f"[{i+1}/{len(picked)}] SKIP id={cid!r} (нет валидной цитаты в чанке)", flush=True)
            errors += 1
            continue

        rq = ok_payload["reference_quote"]
        gold_obj = {
            "question": ok_payload["question"],
            "reference_quote": rq,
            "references": [rq],
            "answer_ref": ok_payload["answer_ref"],
        }
        gold_lines.append(json.dumps(gold_obj, ensure_ascii=False))
        fix_obj = {"id": cid, "document": doc, "metadata": meta}
        fix_lines.append(json.dumps(fix_obj, ensure_ascii=False))
        print(f"[{len(gold_lines)}/{args.n}] OK id={cid}", flush=True)

    if len(gold_lines) < 5:
        print("Собрано слишком мало валидных строк, выход без записи.", flush=True)
        return 4

    fix_path = _ROOT / "eval" / "fixtures" / "chunks_sample.jsonl"
    gold_path = _ROOT / "eval" / "eval_set_gold_v2.jsonl"
    meta_path = _ROOT / "eval" / "eval_set_gold_v2.meta.json"

    fix_path.parent.mkdir(parents=True, exist_ok=True)
    fix_path.write_text("\n".join(fix_lines) + "\n", encoding="utf-8")
    gold_path.write_text("\n".join(gold_lines) + "\n", encoding="utf-8")

    meta = {
        "schema": "gold_v2",
        "version": 2,
        "line_count": len(gold_lines),
        "corpus_type": "chroma_curated",
        "corpus_type_note": "Вопросы и ответы сгенерированы LLM по реальным чанкам Chroma; reference_quote — дословная подстрока чанка (проверено скриптом).",
        "chroma_collection": args.collection,
        "embed_model": embed,
        "build_script": "eval/build_gold_v2_from_chroma.py",
        "built_unix": int(time.time()),
        "fixture_chunks": "eval/fixtures/chunks_sample.jsonl",
        "skipped_chunks": errors,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(gold_lines)} gold rows -> {gold_path.relative_to(_ROOT)}")
    print(f"Wrote {len(fix_lines)} fixture chunks -> {fix_path.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
