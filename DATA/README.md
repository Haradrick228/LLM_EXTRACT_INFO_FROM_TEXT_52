# DATA — корпус, индексы, sidecar

| Путь | Содержимое |
|------|------------|
| `runs/corpus_full_export_20260512/` | Экспорт Chroma → `chunks_from_chroma.jsonl` (**61 543** строк) |
| `runs/phase_b_index_full/<timestamp>/` | Манифесты GPU-индексации dense-коллекций (`manifest.json`, `bases/*.json`) |
| `runs/phase_b_index/` | Индексация gold-fixture баз |
| `indexes/bm25/corpus_full_20260512/` | BM25 sidecar для full corpus |
| `cache/` | Локальные кэши (HyDE и др.) |

Chroma-коллекции живут в сервере Chroma (`CHROMA_HOST` / `CHROMA_PORT`), не в этой папке.
