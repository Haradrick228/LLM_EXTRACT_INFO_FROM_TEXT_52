"""Shared RAG indexing / chunking defaults (legacy script: archive/legacy_indexing/migrate_to_chroma.py)."""
from __future__ import annotations

from typing import List

# Primary chunking used in historical indexing experiments
CHUNK_SIZE = 650
CHUNK_OVERLAP = 250

# RecursiveCharacterTextSplitter separators (same order as archive/legacy_indexing/migrate_to_chroma.py)
CHUNK_SEPARATORS: List[str] = ["\n\n", "\n", "。", "!", "?", "]】", ")", "}", "›"]
