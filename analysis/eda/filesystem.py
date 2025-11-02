from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("analysis.eda.filesystem")


def collect_default_input_dirs(root: Path) -> List[Path]:
    dirs: List[Path] = []
    for candidate in ("scraped_site", "gigachat_materials"):
        path = root / candidate
        if path.is_dir():
            dirs.append(path)
    dirs.extend(p for p in root.iterdir() if p.is_dir() and p.name.startswith("gigachat_"))
    return dirs


def iter_documents(directories: Sequence[Path], extensions: Sequence[str]) -> Iterable[Tuple[Path, Path]]:
    for base_dir in directories:
        if not base_dir.is_dir():
            continue
        for path in base_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                yield base_dir, path


def ensure_write(path: Optional[Path], text: str, overwrite: bool) -> None:
    if path is None:
        return
    if path.exists() and not overwrite:
        LOGGER.debug("Skip writing %s (overwrite disabled).", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_terms(path: Optional[Path]) -> List[str]:
    if not path:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Не удалось найти файл со списком терминов: {path}")
    terms: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        term = line.strip()
        if term:
            terms.append(term)
    return terms
