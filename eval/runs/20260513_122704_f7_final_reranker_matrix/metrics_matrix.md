# F7 final reranker matrix

Run directory: `C:\IT\VS Code\LLM_EXTRACT_INFO_FROM_TEXT_52_DEV\eval\runs\20260513_122704_f7_final_reranker_matrix`

Base retriever: `{'kind': 'hybrid_rrf', 'base_id': 'full_mpnet', 'collection': 'corpus_phase_b_mpnet', 'bm25_dir': 'DATA\\indexes\\bm25\\corpus_full_20260512', 'rrf_k': 60}`
candidate_top_k = `50`; top_k = `20`

## recall@5

| reranker | recall@5 |
|---|---|
| __no_rerank__ | 0.0347 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.0050 |
| BAAI/bge-reranker-v2-m3 | 0.0000 |
| jinaai/jina-reranker-v2-base-multilingual | 0.0347 |

## recall@10

| reranker | recall@10 |
|---|---|
| __no_rerank__ | 0.0347 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.0050 |
| BAAI/bge-reranker-v2-m3 | 0.0050 |
| jinaai/jina-reranker-v2-base-multilingual | 0.0347 |

## recall@20

| reranker | recall@20 |
|---|---|
| __no_rerank__ | 0.0545 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.0050 |
| BAAI/bge-reranker-v2-m3 | 0.0149 |
| jinaai/jina-reranker-v2-base-multilingual | 0.0545 |

## precision@5

| reranker | precision@5 |
|---|---|
| __no_rerank__ | 0.0158 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.0099 |
| BAAI/bge-reranker-v2-m3 | 0.0000 |
| jinaai/jina-reranker-v2-base-multilingual | 0.0158 |

## precision@10

| reranker | precision@10 |
|---|---|
| __no_rerank__ | 0.0139 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.0099 |
| BAAI/bge-reranker-v2-m3 | 0.0010 |
| jinaai/jina-reranker-v2-base-multilingual | 0.0139 |

## mrr

| reranker | mrr |
|---|---|
| __no_rerank__ | 0.0256 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.0099 |
| BAAI/bge-reranker-v2-m3 | 0.0021 |
| jinaai/jina-reranker-v2-base-multilingual | 0.0256 |

## ndcg@3

| reranker | ndcg@3 |
|---|---|
| __no_rerank__ | 0.0161 |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.0099 |
| BAAI/bge-reranker-v2-m3 | 0.0000 |
| jinaai/jina-reranker-v2-base-multilingual | 0.0161 |
