from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Sequence

from .config import CHUNK_OVERLAP, CHUNK_SIZE

LOGGER = logging.getLogger("analysis.eda.text")

try:
    from langdetect import DetectorFactory, LangDetectException, detect_langs
except ImportError:
    DetectorFactory = None
    detect_langs = None

    class LangDetectException(Exception):
        ...

if DetectorFactory is not None:
    DetectorFactory.seed = 0


class TextProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def clean(self, text: str) -> str:
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.replace("\ufeff", " ")
        normalized = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]", " ", normalized)
        normalized = re.sub(r"[^\S\n]+", " ", normalized)
        normalized = re.sub(r"\s+\n", "\n", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r"[ \t]+$", "", normalized, flags=re.MULTILINE)
        return normalized.strip()

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"(?u)\b\w[\w\-]{1,}\b", text.lower())

    def detect_language(self, text: str) -> str:
        if detect_langs is None:
            return "unknown"
        snippet = text[:4000]
        try:
            languages = detect_langs(snippet)
        except LangDetectException:
            return "unknown"
        if not languages:
            return "unknown"
        best = max(languages, key=lambda item: item.prob)
        if best.prob < 0.8:
            return "unknown"
        return best.lang

    def split_into_chunks(self, text: str) -> List[str]:
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        end = len(text)
        while start < end:
            finish = min(start + self.chunk_size, end)
            chunks.append(text[start:finish])
            if finish == end:
                break
            start = max(finish - self.chunk_overlap, start + 1)
        return chunks

    @staticmethod
    def has_non_ascii(text: str) -> bool:
        return any(ord(ch) > 127 for ch in text)


def naive_percentile(seq: Sequence[float], perc: float) -> float:
    if not seq:
        return 0.0
    size = len(seq)
    if size == 1:
        return float(seq[0])
    rank = (size - 1) * perc
    low = int(rank)
    high = min(low + 1, size - 1)
    weight = rank - low
    return float(seq[low] * (1.0 - weight) + seq[high] * weight)
