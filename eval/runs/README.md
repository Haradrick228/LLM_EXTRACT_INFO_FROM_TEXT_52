# Артефакты eval-прогонов (`eval/runs/`)

Immutable выходы retrieval / answer-eval / validity. Каталоги создаются CLI (`run_retrieval_matrix`, diagnostics и т.д.).

## Канонические пути (2026-05)

| Роль | `run_dir` |
|------|-----------|
| B3 full-corpus 6×3 (не менять) | `eval/runs/20260527_161527_b3_full_corpus_rerank` |
| USER-bge-m3 supplement 1×3 (k=5,10,20) | `eval/runs/20260531_012703_b3_full_corpus_rerank` |
| Answer-eval supplement | `eval/runs/final_research_answer_user_bge_m3.json` |
| Answer-eval FRIDA×3 / top8 gap | `eval/runs/final_research_answer_top.json`, `final_research_answer_top8_gap.json` |

Протокол: `eval/config/final_research_protocol.yaml` (`canonical_artifacts`, `supplemental_b3`).

## Индексация (манифесты в `DATA/`)

USER-bge-m3 index: `DATA/runs/phase_b_index_full/20260530_220951/`

## Устаревшие supplement-прогоны

Не использовать в protocol / отчётах (заменены `20260531_012703`):

- `20260530_222344_b3_full_corpus_rerank` — без `recall@20`
- `20260530_222241_b3_full_corpus_rerank` — незавершённый дубль

Удалять только если артефакты уже скопированы в канонический supplement run.
