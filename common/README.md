# `common/` — общие утилиты

Маленькие модули без внешних зависимостей помимо стандартной библиотеки и
`pyyaml`. Импортируются и из `core/`, и из `eval/`, и из `tests/`.

| Модуль | Назначение |
|---|---|
| `structured_logging.py` | `get_logger(name)`, `log_event`, `timed` — единый text/JSON логгер. Формат выбирается через `RAG_LOG_FORMAT={text,json}`. |
| `reproducibility_seeds.py` | `set_global_seed(0)` для Python random / numpy / torch. |
| `environment_snapshot.py` | `git_head_sha`, `pip_freeze_text`, `env_snapshot`, `file_sha256`, `build_environment_manifest`. |
| `project_paths.py` | абсолютные пути: `project_root()`, `datasets_dir()`, `config_dir()`, `runs_dir()`. |
| `config.py` | загрузка конфигов сервиса (исторический). |
| `rag_constants.py` | константы стека (имена коллекций, дефолтные модели). |
| `rag_profile_env.py` | применение профилей `eval/config/rag_profiles.yaml` к `os.environ`. |

Правила:

- Любая I/O‑операция в `common/` должна быть детерминированной и
  безопасной к отсутствию сети / GPU.
- Никаких импортов из `core/` или `eval/` (`common/` должен быть листовым).
