# Tests

Структура зеркалит основное дерево:

| Каталог | Что внутри | Маркер pytest |
|---|---|---|
| `unit/` | быстрые проверки одной функции/модуля, без сети, без Chroma, без GPU | `unit` |
| `integration/` | сценарии с живой Chroma/Ollama (включаются `CHROMA_INTEGRATION=1`) | `integration` |
| `regression/` | замороженные числовые ожидания (replay) | `regression` |
| `fixtures/` | shared фикстуры (заглушки моделей, мини‑корпуса) | — |
| `conftest.py` | глобальный seed=0, регистрация маркеров, fixture-флаги `chroma_available`/`ollama_available` | — |

## Запуск

```bash
# unit-набор: < 10 секунд, без внешних зависимостей
python -m pytest tests/unit

# integration: требует поднятой Chroma на 127.0.0.1:18000
CHROMA_INTEGRATION=1 python -m pytest tests/integration

# regression (replay стабильных метрик):
python -m pytest tests/regression -m regression

# с пропуском медленных:
python -m pytest -m "not integration and not gpu"
```

## Маркеры

Автоматически применяются по каталогу размещения теста (см. `conftest.py::pytest_collection_modifyitems`). Дополнительно есть:

- `gpu` — тесты, которым нужна CUDA / pynvml;
- `slow` — legacy-алиас для regression / integration.

## Naming

`tests/<scope>/test_<что_проверяется>.py`. Имя файла должно отражать SUT, а не
internal-имена утилит. Пример: `test_cross_encoder_reranker.py`, а не
`test_rerank_registry.py`.
