# Metrics ladder (funnel)

- Trace: `C:/IT/VS Code/LLM_EXTRACT_INFO_FROM_TEXT_52_DEV/eval/runs/20260513_205851_gold_on_full_corpus_bge_m3/trace.jsonl`
- Gold: `C:/IT/VS Code/LLM_EXTRACT_INFO_FROM_TEXT_52_DEV/eval/datasets/eval_set_gold_v2.jsonl`
- Corpus oracle file: `C:/IT/VS Code/LLM_EXTRACT_INFO_FROM_TEXT_52_DEV/DATA/runs/corpus_full_export_20260512/chunks_from_chroma.jsonl`
- Hit: **substring**, k_primary=**10**
- Queries: **101**

## Class mix (A=gold not in corpus text, B=not in pool, C=in pool but miss@k, D=hit@k)

| class | count | fraction |
|-------|------:|---------:|
| A | 60 | 0.5941 |
| B | 33 | 0.3267 |
| C | 2 | 0.0198 |
| D | 6 | 0.0594 |
| A_skip | 0 | 0.0000 |

## Wilson 95% CI for hit@k (post-pipeline, binary per query)

- k=10, successes=6, CI≈[0.0275, 0.1236]

## Conditional on oracle_pool (gold hit somewhere in pre-fetch_k)

{
  "n": 8,
  "mean_recall_at_k_post": 0.5625,
  "mean_mrr_post": 0.5574675324675324
}

## Rerank deltas (only traces with stage_02)

{
  "n_traces_with_stage2": 0,
  "mean_delta_mrr": null,
  "mean_delta_recall_at_k": null,
  "mean_ambiguous_rate": null
}
