# HTML-отчёты (`analysis/reports/`)

| Файл | Назначение |
|------|------------|
| [final_canonical_experiment_2026_05.html](final_canonical_experiment_2026_05.html) | Главный отчёт full-corpus (7×3 + USER-bge-m3 supplement). Пересобирается: `python -m eval.reporting.emit_final_canonical_experiment_report --write-html` |
| [final_canonical_experiment_2026_05_raw.html](final_canonical_experiment_2026_05_raw.html) | Снимок HTML **до** cleanup (вкладки без бенчмарка, с warn «6.4% ≠ 64%»). Не перезаписывается emit-скриптом |
| [embedding_benchmark_nagruzka_2026_05.html](embedding_benchmark_nagruzka_2026_05.html) | Бенчмарк embedder: скорость, память, STS-B (CPU/MPS). Встроен во вкладку «Бенчмарк embedder» канонического отчёта |
| [phase_b_final_research_2026_05.html](phase_b_final_research_2026_05.html) | Phase B final research (emit_phase_b_final_research_report) |
| [retrieval_recovery_2026_05.html](retrieval_recovery_2026_05.html) | Retrieval recovery F1–F7 |
| [eda_dashboard.html](eda_dashboard.html) | EDA dashboard |
| [rag_observability_error_analysis_2026_05.html](rag_observability_error_analysis_2026_05.html) | Наблюдаемость RAG + пилот type1/type2 (куратор). `python -m eval.reporting.emit_rag_observability_report --write-html` |

Открывать канонический отчёт из этой папки (относительные ссылки на iframe бенчмарка).
