import argparse
import json
import os
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple

from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:
    SentenceTransformer = None
    CrossEncoder = None


def connect_client(embed_model: str):
    host = os.getenv("CHROMA_HOST", "127.0.0.1")
    port = int(os.getenv("CHROMA_PORT", "18000"))
    tenant = os.getenv("CHROMA_TENANT", "default_tenant")
    database = os.getenv("CHROMA_DATABASE", "default_database")
    embed_fn = SentenceTransformerEmbeddingFunction(model_name=embed_model)
    col_name = os.getenv("CHROMA_COLLECTION", "default")
    client = HttpClient(
        host=host,
        port=port,
        settings=Settings(allow_reset=False, anonymized_telemetry=False),
        tenant=tenant,
        database=database,
    )
    col = client.get_collection(col_name, embedding_function=embed_fn)
    return col


def load_eval(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    random.shuffle(rows)
    return rows


def text_match_score(candidates: List[str], reference: str) -> float:
    ref = reference.lower()
    best = 0.0
    for cand in candidates:
        c = (cand or "").lower()
        if not c:
            continue
        if ref in c or c in ref:
            return 1.0
        overlap = len(set(ref.split()) & set(c.split()))
        if overlap:
            best = max(best, overlap / max(1, len(ref.split())))
    return best


def recall_at_k(found: List[str], refs: List[str], k: int) -> float:
    hits = 0
    for r in refs:
        if any(r.lower() in f.lower() or f.lower() in r.lower() for f in found[:k]):
            hits += 1
    return hits / max(1, len(refs))


def precision_at_k(found: List[str], refs: List[str], k: int) -> float:
    hits = 0
    for f in found[:k]:
        if any(r.lower() in f.lower() or f.lower() in r.lower() for r in refs):
            hits += 1
    return hits / max(1, k)


def mrr(found: List[str], refs: List[str]) -> float:
    refs_l = [r.lower() for r in refs]
    for i, f in enumerate(found):
        fl = f.lower()
        if any(r in fl or fl in r for r in refs_l):
            return 1.0 / (i + 1)
    return 0.0


def ndcg(found: List[str], refs: List[str], k: int) -> float:
    """Бинарная релевантность; DCG с делителем log2(rank+1)."""
    dcg = 0.0
    for i, f in enumerate(found[:k]):
        gain = 1.0 if any(r.lower() in f.lower() or f.lower() in r.lower() for r in refs) else 0.0
        if gain:
            dcg += gain / (1.0 if i == 0 else (math.log2(i + 2)))
    ideal_hits = min(len(refs), k)
    idcg = sum(1.0 / (1.0 if i == 0 else math.log2(i + 2)) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def build_reranker(model_name: str, mode: str):
    if not model_name:
        return None, "none"
    wants_cross = "cross-encoder" in model_name or mode == "cross"
    if wants_cross and CrossEncoder is not None:
        try:
            return CrossEncoder(model_name), "cross"
        except Exception:
            pass
    if SentenceTransformer is None:
        return None, "none"
    try:
        return SentenceTransformer(model_name), "bi"
    except Exception:
        return None, "none"


def rerank(question: str, docs: List[str], model, mode: str) -> List[Tuple[str, float]]:
    if not model:
        return [(d, 0.0) for d in docs]
    try:
        if mode == "cross":
            pairs = [(question, d) for d in docs]
            scores = model.predict(pairs).tolist()
        else:
            q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
            d_emb = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
            scores = (d_emb @ q_emb).tolist()
        scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return scored
    except Exception:
        return [(d, 0.0) for d in docs]


def evaluate(col, dataset: List[Dict], k_values: List[int], fetch_k: int, rerank_model=None, rerank_mode: str = "none") -> Dict[str, float]:
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"precision@{k}": [] for k in k_values})
    metrics["mrr"] = []
    metrics["ndcg@3"] = []

    for row in dataset:
        q = row["question"]
        refs = row.get("references") or []
        res = col.query(query_texts=[q], n_results=fetch_k, include=["documents"])
        docs = (res or {}).get("documents", [[]])[0] if res else []
        docs = [d or "" for d in docs]
        if rerank_model:
            docs = [d for d, _ in rerank(q, docs, rerank_model, rerank_mode)]

        for k in k_values:
            metrics[f"recall@{k}"].append(recall_at_k(docs, refs, k))
            metrics[f"precision@{k}"].append(precision_at_k(docs, refs, k))
        metrics["mrr"].append(mrr(docs, refs))
        metrics["ndcg@3"].append(ndcg(docs, refs, 3))

    out = {}
    for k, vals in metrics.items():
        out[k] = sum(vals) / max(1, len(vals))
    return out


def main():
    parser = argparse.ArgumentParser(description="Retrieval evaluation for Chroma-backed RAG.")
    parser.add_argument("--eval-file", required=True, help="Path to eval_set.jsonl")
    parser.add_argument("--embed-model", default=os.getenv("CHROMA_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"))
    parser.add_argument("--k", default="5,10", help="Comma-separated k values, e.g. 5,10")
    parser.add_argument("--fetch-k", type=int, default=int(os.getenv("RETRIEVER_FETCH_K", "50") or 50), help="How many docs to fetch before rerank")
    parser.add_argument("--rerank-model", default=os.getenv("RERANK_MODEL", ""), help="Optional rerank model (CrossEncoder or bi-encoder)")
    parser.add_argument("--rerank-mode", default=os.getenv("RERANK_MODE", "auto"), help="auto|cross|bi")
    args = parser.parse_args()

    k_values = [int(x) for x in args.k.split(",") if x.strip()]
    dataset = load_eval(Path(args.eval_file))
    col = connect_client(args.embed_model)
    rerank_model, rerank_mode = build_reranker(args.rerank_model, args.rerank_mode)
    scores = evaluate(col, dataset, k_values, fetch_k=args.fetch_k, rerank_model=rerank_model, rerank_mode=rerank_mode)
    for k, v in scores.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
