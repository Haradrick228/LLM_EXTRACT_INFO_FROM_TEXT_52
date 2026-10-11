# F4 BGE-M3 multi-vector sweep

Run directory: `C:\IT\VS Code\LLM_EXTRACT_INFO_FROM_TEXT_52_DEV\eval\runs\20260513_020808_f4_bge_m3_multi_vector_sweep`

Candidate pool top-k = `100`; metric pool top-k = `20`

## Weights

| combo | w_dense | w_sparse | w_colbert |
|---|---|---|---|
| dense_only | 1.0 | 0.0 | 0.0 |
| dense_sparse | 0.6 | 0.4 | 0.0 |
| dense_colbert | 0.5 | 0.0 | 0.5 |
| balanced | 0.4 | 0.2 | 0.4 |
| sparse_heavy | 0.2 | 0.6 | 0.2 |

## recall@5

| combo | recall@5 |
|---|---|
| dense_only | 0.0347 |
| dense_sparse | 0.0446 |
| dense_colbert | 0.0545 |
| balanced | 0.0545 |
| sparse_heavy | 0.0446 |

## recall@10

| combo | recall@10 |
|---|---|
| dense_only | 0.0446 |
| dense_sparse | 0.0545 |
| dense_colbert | 0.0644 |
| balanced | 0.0545 |
| sparse_heavy | 0.0545 |

## recall@20

| combo | recall@20 |
|---|---|
| dense_only | 0.0446 |
| dense_sparse | 0.0545 |
| dense_colbert | 0.0644 |
| balanced | 0.0644 |
| sparse_heavy | 0.0644 |

## precision@5

| combo | precision@5 |
|---|---|
| dense_only | 0.0198 |
| dense_sparse | 0.0257 |
| dense_colbert | 0.0238 |
| balanced | 0.0277 |
| sparse_heavy | 0.0317 |

## precision@10

| combo | precision@10 |
|---|---|
| dense_only | 0.0228 |
| dense_sparse | 0.0248 |
| dense_colbert | 0.0257 |
| balanced | 0.0248 |
| sparse_heavy | 0.0228 |

## mrr

| combo | mrr |
|---|---|
| dense_only | 0.0432 |
| dense_sparse | 0.0452 |
| dense_colbert | 0.0470 |
| balanced | 0.0473 |
| sparse_heavy | 0.0462 |

## ndcg@3

| combo | ndcg@3 |
|---|---|
| dense_only | 0.0319 |
| dense_sparse | 0.0350 |
| dense_colbert | 0.0319 |
| balanced | 0.0350 |
| sparse_heavy | 0.0350 |
