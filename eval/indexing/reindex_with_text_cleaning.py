"""
Reindex current Chroma collection into a clean deduplicated one.

Rules:
- Source collection: CHROMA_COLLECTION (default: "default")
- Target collection: CHROMA_COLLECTION_CLEAN (default: "default_clean")
- Embed model: CHROMA_EMBED_MODEL or sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- Skips docs with len(text.strip()) < 30
- Dedup key: (source, page_label, page, md5(text))
"""

import hashlib
import os
from typing import Dict, List, Set, Tuple

from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


def connect_collection(name: str, embed_model: str):
    host = os.getenv("CHROMA_HOST", "127.0.0.1")
    port = int(os.getenv("CHROMA_PORT", "18000"))
    tenant = os.getenv("CHROMA_TENANT", "default_tenant")
    database = os.getenv("CHROMA_DATABASE", "default_database")
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)
    client = HttpClient(
        host=host,
        port=port,
        settings=Settings(allow_reset=False, anonymized_telemetry=False),
        tenant=tenant,
        database=database,
    )
    try:
        return client.get_collection(name, embedding_function=embed_fn)
    except Exception:
        return client.create_collection(name, embedding_function=embed_fn)


def fetch_batch(col, offset: int, limit: int):
    return col.get(include=["metadatas", "documents"], offset=offset, limit=limit)


def fetch_ids(col, batch_size: int = 5000) -> Set[str]:
    total = col.count()
    ids: Set[str] = set()
    for offset in range(0, total, batch_size):
        res = col.get(offset=offset, limit=batch_size, include=[])
        ids.update(res.get("ids") or [])
    return ids


def main():
    src_name = os.getenv("CHROMA_COLLECTION", "default")
    dst_name = os.getenv("CHROMA_COLLECTION_CLEAN", "default_clean")
    embed_model = os.getenv(
        "CHROMA_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    src = connect_collection(src_name, embed_model)
    dst = connect_collection(dst_name, embed_model)

    total = src.count()
    print(f"Source collection '{src_name}' documents: {total}")
    existing_ids = fetch_ids(dst)
    if existing_ids:
        print(f"Target already has {len(existing_ids)} docs; will skip them (resume mode).")

    seen: Set[Tuple[str, str, str, str]] = set()
    batch_size = 500
    inserted = 0

    for offset in range(0, total, batch_size):
        res = fetch_batch(src, offset=offset, limit=batch_size)
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        docs = res.get("documents") or []

        upsert_ids: List[str] = []
        upsert_docs: List[str] = []
        upsert_metas: List[Dict] = []

        for doc_id, meta, doc in zip(ids, metas, docs):
            if doc_id in existing_ids:
                continue
            text = (doc or "").strip()
            if len(text) < 30:
                continue
            sig = (
                str(meta.get("source")),
                str(meta.get("page_label")),
                str(meta.get("page")),
                hashlib.md5(text.encode("utf-8")).hexdigest(),
            )
            if sig in seen:
                continue
            seen.add(sig)
            upsert_ids.append(doc_id)
            upsert_docs.append(text)
            upsert_metas.append(meta or {})

        if upsert_ids:
            dst.upsert(ids=upsert_ids, documents=upsert_docs, metadatas=upsert_metas)
            inserted += len(upsert_ids)
            print(f"Offset {offset}: upserted {len(upsert_ids)}, total inserted {inserted}")

    print(f"Done. Inserted {inserted} unique docs into '{dst_name}'.")


if __name__ == "__main__":
    main()
