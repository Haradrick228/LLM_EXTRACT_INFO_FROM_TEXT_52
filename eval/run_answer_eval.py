import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

# allow running as script from repo root
sys.path.append(str(Path(".").resolve()))

from core.rag.rag_service import RagService


def load_dataset(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def connect_chroma(embed_model: str):
    # Clear env vars that conflict with HttpClient
    for key in list(os.environ.keys()):
        if key.startswith("CHROMA_") and key not in ("CHROMA_COLLECTION", "CHROMA_EMBED_MODEL", "ANSWER_REF_EMBED_MODEL"):
            del os.environ[key]

    host = "127.0.0.1"
    port = 18000
    tenant = "default_tenant"
    database = "default_database"
    col_name = os.getenv("CHROMA_COLLECTION", "test_frida")
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)

    # Override .env settings
    os.environ["CHROMA_SERVER_HOST"] = host
    os.environ["CHROMA_SERVER_HTTP_PORT"] = str(port)

    client = HttpClient(
        host=host,
        port=port,
        settings=Settings(allow_reset=False, anonymized_telemetry=False),
        tenant=tenant,
        database=database,
    )
    return client.get_collection(col_name, embedding_function=embed_fn)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    dataset_path = Path(os.getenv("ANSWER_EVAL_FILE", "eval/eval_set_with_answer_ref.jsonl"))
    out_path = Path(os.getenv("ANSWER_EVAL_OUT", "eval/answer_eval_results.jsonl"))
    fetch_k = int(os.getenv("ANSWER_EVAL_FETCH_K", os.getenv("RETRIEVER_FETCH_K", "40")))
    embed_model = os.getenv("CHROMA_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ref_embed_model = os.getenv("ANSWER_REF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Override .env settings for RagService and chromadb
    for key in list(os.environ.keys()):
        if key.startswith("CHROMA_"):
            del os.environ[key]
    os.environ["CHROMA_HOST"] = "127.0.0.1"
    os.environ["CHROMA_PORT"] = "18000"
    os.environ["CHROMA_COLLECTION"] = os.getenv("CHROMA_COLLECTION", "default")
    os.environ["CHROMA_EMBED_MODEL"] = embed_model
    os.environ["CHROMA_SERVER_HOST"] = "127.0.0.1"
    os.environ["CHROMA_SERVER_HTTP_PORT"] = "18000"

    data = load_dataset(dataset_path)
    rag = RagService()
    chroma_col = connect_chroma(embed_model)
    ref_embedder: Optional[SentenceTransformer] = None
    if SentenceTransformer is not None:
        try:
            ref_embedder = SentenceTransformer(ref_embed_model)
        except Exception:
            ref_embedder = None

    agg_ref_sim = []
    agg_refs_sim = []
    results = []

    for row in data:
        q = row["question"]
        refs = row.get("references") or []
        answer_ref = row.get("answer_ref")

        # generate answer
        ans = rag.answer(q)

        # retrieve context
        docs = []
        try:
            res = chroma_col.query(query_texts=[q], n_results=fetch_k, include=["documents"])
            docs = (res or {}).get("documents", [[]])[0] if res else []
        except Exception:
            docs = []

        ref_sim = None
        refs_sim = None
        if ref_embedder is not None:
            try:
                ans_emb = ref_embedder.encode([ans], convert_to_numpy=True, normalize_embeddings=True)[0]
                if answer_ref:
                    ref_emb = ref_embedder.encode([answer_ref], convert_to_numpy=True, normalize_embeddings=True)[0]
                    ref_sim = cosine(ans_emb, ref_emb)
                    agg_ref_sim.append(ref_sim)
                if refs:
                    refs_emb = ref_embedder.encode(refs, convert_to_numpy=True, normalize_embeddings=True)
                    sims = [cosine(ans_emb, r) for r in refs_emb]
                    if sims:
                        refs_sim = max(sims)
                        agg_refs_sim.append(refs_sim)
            except Exception:
                pass

        results.append(
            {
                "question": q,
                "answer": ans,
                "answer_ref": answer_ref,
                "ref_sim": ref_sim,
                "refs_sim": refs_sim,
                "refs_count": len(refs),
                "docs_sample": docs[:3],
            }
        )

    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results), encoding="utf-8")

    metrics = {
        "avg_ref_sim": float(np.mean(agg_ref_sim)) if agg_ref_sim else None,
        "avg_refs_sim": float(np.mean(agg_refs_sim)) if agg_refs_sim else None,
        "count_ref": len(agg_ref_sim),
        "count_refs": len(agg_refs_sim),
    }
    print("Answer eval done. Metrics:", metrics)
    print("Saved results to", out_path)


if __name__ == "__main__":
    main()
