# Eval — исследовательская документация

Долгоживущие заметки и отчёты Phase B (не генерируются автоматически при каждом прогоне).

| Документ | Тема |
|----------|------|
| [METRICS_LADDER.md](METRICS_LADDER.md) | Лестница метрик A–D, Wilson |
| [PHASE_B3_FULL_CORPUS_RERANK_PLAN.md](PHASE_B3_FULL_CORPUS_RERANK_PLAN.md) | План B3 6×3 на 61k |
| [PHASE_B_RESEARCH_METRICS_REPORT.md](PHASE_B_RESEARCH_METRICS_REPORT.md) | Сводка исследования Phase B |
| [RETRIEVAL_FULL_CORPUS_RESULTS_2026_05.md](RETRIEVAL_FULL_CORPUS_RESULTS_2026_05.md) | F2–F7 / full corpus |
| [RETRIEVAL_DIAGNOSIS_2026_05.md](RETRIEVAL_DIAGNOSIS_2026_05.md) | Диагноз низкого recall |
| [RETRIEVAL_WORST_QUERIES_2026_05.md](RETRIEVAL_WORST_QUERIES_2026_05.md) | Худшие запросы |
| [ANSWER_EVAL_SUMMARY.md](ANSWER_EVAL_SUMMARY.md) | Answer-eval сводка |
| [answer_eval_plan.md](answer_eval_plan.md) | План answer-eval |
| [rerank_results.md](rerank_results.md) | Ранние rerank-результаты |
| [RAG_CLEANUP_PLAN.md](RAG_CLEANUP_PLAN.md) | План очистки RAG/eval |

HTML (наблюдаемость / ошибки type1–type2): [../analysis/reports/rag_observability_error_analysis_2026_05.html](../analysis/reports/rag_observability_error_analysis_2026_05.html) — `python -m eval.reporting.emit_rag_observability_report --write-html [--run-classify]`.

Вход в модуль: [../README.md](../README.md). Handoff агентам: [../PHASE_B_SCOPE.md](../PHASE_B_SCOPE.md).  
Генерируемый канонический отчёт: [../FINAL_CANONICAL_EXPERIMENT_REPORT_2026_05.md](../FINAL_CANONICAL_EXPERIMENT_REPORT_2026_05.md).
