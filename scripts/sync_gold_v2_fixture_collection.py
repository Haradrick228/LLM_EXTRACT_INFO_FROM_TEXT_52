"""
Создаёт/обновляет коллекцию Chroma `gold_v2_fixture` из eval/fixtures/chunks_sample.jsonl
(те же тексты, что в eval_set_gold_v2), чтобы retrieval-метрики на golden были > 0.

  python scripts/sync_gold_v2_fixture_collection.py

Нужны: Docker Chroma, тот же эмбеддер, что в eval (по умолчанию mpnet, локальная папка DATA/hf_embedders при наличии).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction  # noqa: E402

from eval.chroma_utils import chroma_http_client  # noqa: E402

COLLECTION = "gold_v2_fixture"
FIXTURE = _ROOT / "eval" / "fixtures" / "chunks_sample.jsonl"


def _embed_model() -> str:
    local = _ROOT / "DATA" / "hf_embedders" / "sentence-transformers__paraphrase-multilingual-mpnet-base-v2"
    if (local / "config.json").is_file():
        return str(local)
    return os.getenv(
        "CHROMA_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )


def main() -> int:
    os.environ.setdefault("RAG_EVAL_CHROMA_HOST", "127.0.0.1")
    os.environ.setdefault("RAG_EVAL_CHROMA_PORT", "18000")

    embed = _embed_model()
    ef = SentenceTransformerEmbeddingFunction(model_name=embed)
    client = chroma_http_client()

    try:
        client.delete_collection(COLLECTION)
        print(f"[sync_gold_v2] удалена коллекция {COLLECTION}", flush=True)
    except Exception as e:
        print(f"[sync_gold_v2] delete (можно игнорировать): {e}", flush=True)

    try:
        col = client.create_collection(
            name=COLLECTION,
            embedding_function=ef,
            metadata={"source": "eval/fixtures/chunks_sample.jsonl", "gold_schema": "v2"},
        )
    except Exception as e:
        print(
            "[sync_gold_v2] не удалось создать коллекцию (часто Chroma volume readonly или права на /data). "
            f"Ошибка: {e}\n"
            "Можно прогонять gold_v2 на коллекции `default`, если там есть те же тексты.",
            flush=True,
        )
        return 2

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids.append(row["id"])
        docs.append(row["document"])
        metas.append(row.get("metadata") or {})

    col.add(ids=ids, documents=docs, metadatas=metas)
    print(f"[sync_gold_v2] коллекция={COLLECTION} документов={col.count()} embed={embed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
