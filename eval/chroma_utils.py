"""Shared Chroma HTTP client helpers for eval / export (avoid copy-paste)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Не удалять из окружения — нужны вызывающему коду / RagService после создания клиента.
_PRESERVE_CHROMA_ENV_KEYS = frozenset(
    {"CHROMA_COLLECTION", "CHROMA_EMBED_MODEL", "CHROMA_TENANT", "CHROMA_DATABASE"}
)


def chroma_http_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    tenant: Optional[str] = None,
    database: Optional[str] = None,
) -> HttpClient:
    host = host or os.getenv("CHROMA_HOST", "127.0.0.1")
    port = int(port or os.getenv("CHROMA_PORT", "18000"))
    tenant = tenant or os.getenv("CHROMA_TENANT", "default_tenant")
    database = database or os.getenv("CHROMA_DATABASE", "default_database")
    for key in list(os.environ.keys()):
        if key.startswith("CHROMA_") and key not in _PRESERVE_CHROMA_ENV_KEYS:
            del os.environ[key]
    os.environ["CHROMA_SERVER_HOST"] = host
    os.environ["CHROMA_SERVER_HTTP_PORT"] = str(port)
    return HttpClient(
        host=host,
        port=port,
        settings=Settings(allow_reset=False, anonymized_telemetry=False),
        tenant=tenant,
        database=database,
    )


def get_collection(
    client: HttpClient,
    name: str,
    embed_model: str,
):
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)
    return client.get_collection(name, embedding_function=embed_fn)


def export_collection_batches(
    col,
    *,
    batch_size: int = 500,
    include: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    include = include or ["metadatas", "documents"]
    total = col.count()
    offset = 0
    while offset < total:
        res = col.get(offset=offset, limit=batch_size, include=include)
        yield res
        offset += batch_size


def export_collection_to_jsonl(
    col,
    out_path: Path,
    *,
    batch_size: int = 500,
) -> int:
    """
    Write one JSON object per line: {"id", "document", "metadata"}.
    Returns number of rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for batch in export_collection_batches(col, batch_size=batch_size, include=["metadatas", "documents"]):
            ids = batch.get("ids") or []
            docs = batch.get("documents") or []
            metas = batch.get("metadatas") or []
            for i, doc_id in enumerate(ids):
                row = {
                    "id": doc_id,
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    return n
