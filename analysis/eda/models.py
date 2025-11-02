from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class PipelineSettings:
    input_dirs: Sequence[Path]
    output_dir: Optional[Path]
    report_dir: Path
    overwrite: bool
    max_files: Optional[int]
    terms: Sequence[str]


@dataclass(frozen=True)
class DocumentRecord:
    source_path: Path
    output_path: Optional[Path]
    content_hash: str
    characters: int
    tokens: int
    chunks: int
    language: str
    has_non_ascii: bool


@dataclass(frozen=True)
class ProcessingError:
    path: str
    error: str


@dataclass(frozen=True)
class CleanMetadataUpdate:
    linked: bool
    available: bool
    message: Optional[str] = None


@dataclass(frozen=True)
class ReportPaths:
    json_report: Path
    html_dashboard: Path
