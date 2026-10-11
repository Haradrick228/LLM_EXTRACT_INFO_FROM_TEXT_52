import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from eval.reporting.log_eval_run_config_banner import write_eval_banner

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

# allow running as script from repo root
sys.path.append(str(Path(".").resolve()))

def load_dataset(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    dataset_path = Path(os.getenv("ANSWER_EVAL_FILE", "eval/datasets/eval_set_with_answer_ref.jsonl"))
    if not dataset_path.exists():
        legacy = Path("eval/eval_set_with_answer_ref.jsonl")
        if legacy.exists():
            dataset_path = legacy
    out_path = Path(os.getenv("ANSWER_EVAL_OUT", "eval/answer_eval_results.jsonl"))
    fetch_k = int(os.getenv("ANSWER_EVAL_FETCH_K", os.getenv("RETRIEVER_FETCH_K", "40")))
    embed_model = os.getenv("CHROMA_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ref_embed_model = os.getenv("ANSWER_REF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    host = os.getenv("RAG_EVAL_CHROMA_HOST", os.getenv("CHROMA_HOST", "127.0.0.1"))
    port = os.getenv("RAG_EVAL_CHROMA_PORT", os.getenv("CHROMA_PORT", "18000"))
    collection = os.getenv("CHROMA_COLLECTION", "default")
    max_rows = int(os.getenv("ANSWER_EVAL_MAX_ROWS", "0") or 0)
    chunking_hint = os.getenv("CHUNKING_LABEL", os.getenv("CHUNK_PROFILE", ""))
    write_eval_banner(
        "run_answer_eval",
        {
            "dataset_path": str(dataset_path),
            "out_path": str(out_path),
            "chroma_embed_model": embed_model,
            "answer_ref_embed_model": ref_embed_model,
            "fetch_k": fetch_k,
            "chroma_host": host,
            "chroma_port": port,
            "chroma_collection": collection,
            "rerank_model": os.getenv("RERANK_MODEL", ""),
            "rerank_mode": os.getenv("RERANK_MODE", ""),
            "rerank_device": os.getenv("RERANK_DEVICE", ""),
            "llm_provider": os.getenv("LLM_PROVIDER", ""),
            "llm_provider_model": os.getenv("LLM_PROVIDER_MODEL", ""),
            "ollama_model": os.getenv("OLLAMA_MODEL", ""),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", ""),
            "rag_profile": os.getenv("RAG_PROFILE", ""),
            "max_rows": max_rows,
            "chunking_label": chunking_hint or "(not set)",
        },
    )
    for key in list(os.environ.keys()):
        if key.startswith("CHROMA_"):
            del os.environ[key]
    os.environ["CHROMA_HOST"] = host
    os.environ["CHROMA_PORT"] = str(port)
    os.environ["CHROMA_COLLECTION"] = collection
    os.environ["CHROMA_EMBED_MODEL"] = embed_model
    os.environ["CHROMA_SERVER_HOST"] = host
    os.environ["CHROMA_SERVER_HTTP_PORT"] = str(port)

    data = load_dataset(dataset_path)
    if max_rows > 0:
        data = data[:max_rows]

    # Import after env is aligned so ChromaConfig / stack read the intended collection and embedder.
    from core.rag.service.rag_service import RagService

    rag = RagService(
        chroma_collection=collection,
        chroma_host=host,
        chroma_port=int(port),
        chroma_embed_model=embed_model,
        graph_enabled=False,
    )
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

        ans = rag.answer(q)

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
