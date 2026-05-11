"""
Скачивание HF-моделей (эмбеддеры + реранкеры) в отдельную папку с видимым прогрессом.

Использует CLI `hf download` (Hugging Face Hub) **без** --quiet — в консоли идут
прогресс-бары по файлам. Для длинных скачиваний запускайте из терминала вручную:

  python scripts/download_embedders.py --out DATA/hf_embedders

После скачивания укажите локальный путь в .env (примеры в конце --help).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Репозитории, которые реально использует текущий код / профили по умолчанию
DEFAULT_REPOS = [
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # CHROMA_EMBED_MODEL, RagService
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # типовой cross-реранкер в проверках
    "sentence-transformers/all-MiniLM-L6-v2",  # eval/rag_profiles baseline_mpnet_rerank, ANSWER_REF
]

# Крупные; матрица метрик — только если нужны профили из rag_profiles.yaml
EXTRA_MATRIX_REPOS = [
    "jinaai/jina-reranker-v3",
    "BAAI/bge-reranker-v2-m3",
    "Qwen/Qwen3-Reranker-4B",
]


def _local_subdir(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _hub_models_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    base = Path(hf_home) if hf_home else Path.home() / ".cache" / "huggingface"
    return base / "hub"


def _repo_cache_folder_name(repo_id: str) -> str:
    org, name = repo_id.split("/", 1)
    return f"models--{org}--{name}"


def _resolve_snapshot_dir(hub: Path, repo_id: str) -> Path | None:
    repo_dir = hub / _repo_cache_folder_name(repo_id)
    if not repo_dir.is_dir():
        return None
    ref_main = repo_dir / "refs" / "main"
    if ref_main.is_file():
        h = ref_main.read_text(encoding="utf-8").strip()
        cand = repo_dir / "snapshots" / h
        if cand.is_dir():
            return cand
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return None
    children = [p for p in snaps.iterdir() if p.is_dir()]
    if not children:
        return None
    return max(children, key=lambda p: p.stat().st_mtime)


def _copy_snapshot_tree_verbose(src: Path, dst: Path) -> int:
    """Копирует файлы снапшота (разыменовывает symlinks на blobs). Печатает каждый файл — видно «скачку» без сети."""
    files: list[tuple[Path, Path]] = []
    for p in src.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            files.append((p, dst / rel))
    n = len(files)
    if n == 0:
        return 0
    for i, (src_f, dst_f) in enumerate(files, 1):
        rel = src_f.relative_to(src)
        dst_f.parent.mkdir(parents=True, exist_ok=True)
        real = src_f.resolve() if src_f.is_symlink() else src_f
        if not real.is_file():
            continue
        mb = real.stat().st_size / (1024 * 1024)
        print(f"  [{i}/{n}] {rel.as_posix()}  ({mb:.2f} MiB)", flush=True)
        shutil.copy2(real, dst_f)
    return n


def main() -> int:
    p = argparse.ArgumentParser(description="HF: скачать эмбеддеры/реранкеры в папку (с прогрессом).")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("DATA/hf_embedders"),
        help="Корневая папка (внутри — подкаталог на каждую модель).",
    )
    p.add_argument(
        "--include-matrix",
        action="store_true",
        help=f"Дополнительно скачать крупные реранкеры: {', '.join(EXTRA_MATRIX_REPOS)}",
    )
    p.add_argument("--max-workers", type=int, default=4, help="Параллельные загрузки для hf download.")
    p.add_argument(
        "--from-cache",
        action="store_true",
        help="Не ходить в интернет: скопировать снапшот из HF hub-кэша (~/.cache/huggingface/hub). "
        "Печатает каждый файл — удобно смотреть прогресс.",
    )
    p.add_argument(
        "--hf-hub",
        type=Path,
        default=None,
        help="Папка hub (по умолчанию: $HF_HOME/hub или ~/.cache/huggingface/hub).",
    )
    args = p.parse_args()

    root: Path = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)

    repos = list(DEFAULT_REPOS)
    if args.include_matrix:
        repos.extend(EXTRA_MATRIX_REPOS)

    hub = (args.hf_hub or _hub_models_dir()).resolve()

    print(f"[download_embedders] корень вывода: {root}", flush=True)
    print(f"[download_embedders] HF hub-кэш: {hub}", flush=True)
    print(f"[download_embedders] режим: {'копия из кэша' if args.from_cache else 'hf download (сеть)'}", flush=True)

    for repo_id in repos:
        target = root / _local_subdir(repo_id)
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 72, flush=True)
        print(f"[download_embedders] repo={repo_id}", flush=True)
        print(f"[download_embedders] local-dir={target}", flush=True)
        print("=" * 72 + "\n", flush=True)

        if args.from_cache:
            snap = _resolve_snapshot_dir(hub, repo_id)
            if snap is None:
                print(
                    f"[download_embedders] В кэше нет { _repo_cache_folder_name(repo_id) } — "
                    f"сначала один раз скачайте с интернета (без --from-cache) или положите снапшот вручную.",
                    flush=True,
                )
                return 2
            n = _copy_snapshot_tree_verbose(snap, target)
            print(f"[download_embedders] скопировано файлов: {n}", flush=True)
            if n == 0:
                return 2
            continue

        # Явно разрешаем сеть (у части машин в профиле висит HF_HUB_OFFLINE=1).
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            env.pop(k, None)
        env["HF_HUB_OFFLINE"] = "0"

        cmd = [
            sys.executable,
            "-m",
            "huggingface_hub.cli.hf",
            "download",
            repo_id,
            "--local-dir",
            str(target),
            "--max-workers",
            str(args.max_workers),
        ]
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"[download_embedders] ОШИБКА код={r.returncode} для {repo_id}", flush=True)
            return r.returncode

    print("\n[download_embedders] готово. Примеры для .env:", flush=True)
    for repo_id in DEFAULT_REPOS + (EXTRA_MATRIX_REPOS if args.include_matrix else []):
        sub = _local_subdir(repo_id)
        pth = root / sub
        key = "CHROMA_EMBED_MODEL" if "paraphrase-multilingual-mpnet" in repo_id else "RERANK_MODEL / ref"
        print(f"  # {repo_id} ({key})", flush=True)
        print(f"  # {pth}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
