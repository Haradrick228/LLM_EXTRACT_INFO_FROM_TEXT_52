import datetime as dt
import hashlib
import json
import math
import os
import pickle
import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, List, Optional

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.orm import sessionmaker
import psycopg2


@dataclass
class CacheHit:
    question: str
    answer: str
    files: List[str]
    model: Optional[str]
    question_hash: str
    answer_hash: str
    created_at: float
    similarity: float = 1.0


class QueryCache:

    def __init__(
        self,
        dsn: str,
        *,
        schema: Optional[str] = None,
        table: str = "cached_answers",
        similarity_scan_limit: int = 200,
    ) -> None:
        if not dsn:
            raise ValueError("PostgreSQL DSN must be provided for QueryCache")

        self.dsn = dsn
        self.schema = self._validate_identifier(schema) if schema else None
        self.table = self._validate_identifier(table or "cached_answers")
        self.full_table = f"{self.schema}.{self.table}" if self.schema else self.table
        self.similarity_scan_limit = max(1, int(similarity_scan_limit))

        self._lock = threading.RLock()
        self._engine = create_engine(
            "postgresql+psycopg2://",
            creator=lambda: psycopg2.connect(dsn=dsn),
            pool_pre_ping=True,
            future=True,
        )
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)
        self._metadata = MetaData()
        self._table = Table(
            self.table,
            self._metadata,
            Column("id", BigInteger, primary_key=True),
            Column("question", Text, nullable=False),
            Column("question_normalized", Text, nullable=False),
            Column("answer", Text, nullable=False),
            Column("answer_normalized", Text, nullable=False),
            Column("files", JSONB),
            Column("model", String(255)),
            Column("model_key", String(255), nullable=False, server_default=""),
            Column("question_hash", String(64), nullable=False),
            Column("answer_hash", String(64), nullable=False),
            Column("embedding", LargeBinary),
            Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
            schema=self.schema,
        )
        self._uq_index = Index(
            self._index_name("uq_qhash_model"),
            self._table.c.question_hash,
            self._table.c.model_key,
            unique=True,
        )
        self._created_index = Index(
            self._index_name("idx_created_at"),
            self._table.c.created_at.desc(),
        )

        self._ensure_schema_and_table()

    @staticmethod
    def _validate_identifier(name: Optional[str]) -> str:
        if not name:
            raise ValueError("SQL identifier must not be empty")
        ident = name.strip()
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", ident):
            raise ValueError(f"Invalid SQL identifier: {name!r}")
        return ident

    def _index_name(self, suffix: str) -> str:
        base = f"{self.table}_{suffix}"
        base = re.sub(r"[^A-Za-z0-9_]", "_", base)
        if self.schema:
            return f"{self.schema}_{base}"
        return base

    def _ensure_schema_and_table(self) -> None:
        with self._engine.begin() as conn:
            if self.schema and self.schema.lower() != "public":
                conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.schema}"))
        self._metadata.create_all(self._engine, tables=[self._table])
        for idx in (self._uq_index, self._created_index):
            idx.create(self._engine, checkfirst=True)

    @staticmethod
    def normalize_text(text: Optional[str]) -> str:
        if text is None:
            return ""
        lowered = text.strip().lower()
        return re.sub(r"\s+", " ", lowered)

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_embedding(embedding: Optional[Sequence[float]]) -> Optional[bytes]:
        if embedding is None:
            return None
        seq = list(embedding)
        if not seq:
            return None
        return pickle.dumps(seq, protocol=4)

    @staticmethod
    def _deserialize_embedding(payload: Optional[Any]) -> Optional[List[float]]:
        if payload is None:
            return None
        if isinstance(payload, memoryview):
            payload = payload.tobytes()
        if not isinstance(payload, (bytes, bytearray)):
            return None
        try:
            value = pickle.loads(bytes(payload))
        except Exception:
            return None
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return None

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return max(min(dot / (math.sqrt(norm_a) * math.sqrt(norm_b)), 1.0), -1.0)

    @staticmethod
    def _normalize_model(model: Optional[str]) -> str:
        if model is None:
            return ""
        return model.strip().lower()

    @staticmethod
    def _coerce_files(raw: Any) -> List[str]:
        if raw is None:
            return []
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception:
                return []
        elif isinstance(raw, (list, tuple)):
            data = list(raw)
        else:
            return []
        out: List[str] = []
        for item in data:
            if isinstance(item, str) and item:
                out.append(item)
        return out

    @classmethod
    def resolve_dsn_from_env(cls) -> Optional[str]:
        dsn = (os.getenv("LLM_CACHE_DSN") or "").strip()
        if dsn:
            return dsn

        host = os.getenv("LLM_CACHE_HOST") or os.getenv("POSTGRES_HOST")
        port = os.getenv("LLM_CACHE_PORT") or os.getenv("POSTGRES_PORT")
        dbname = os.getenv("LLM_CACHE_DB") or os.getenv("POSTGRES_DB")
        user = os.getenv("LLM_CACHE_USER") or os.getenv("POSTGRES_USER")
        password = os.getenv("LLM_CACHE_PASSWORD") or os.getenv("POSTGRES_PASSWORD")
        sslmode = os.getenv("LLM_CACHE_SSLMODE") or os.getenv("POSTGRES_SSLMODE")
        options = os.getenv("LLM_CACHE_OPTIONS")

        if not all([host, port, dbname, user]):
            return None

        try:
            port_int = int(port)
        except Exception as exc:
            raise ValueError(f"Invalid port for LLM cache: {port!r}") from exc

        parts = [
            f"host={host}",
            f"port={port_int}",
            f"dbname={dbname}",
            f"user={user}",
        ]
        if password is not None:
            parts.append(f"password={password}")
        if sslmode:
            parts.append(f"sslmode={sslmode}")
        if options:
            parts.append(options)
        return " ".join(parts)

    def store(
        self,
        *,
        question: str,
        answer: str,
        files: Optional[Sequence[str]] = None,
        model: Optional[str] = None,
        normalized_question: Optional[str] = None,
        normalized_answer: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        if not question or not answer:
            return

        norm_question = (
            normalized_question if normalized_question is not None else self.normalize_text(question)
        )
        norm_answer = (
            normalized_answer if normalized_answer is not None else self.normalize_text(answer)
        )
        question_hash = self.hash_text(norm_question)
        answer_hash = self.hash_text(norm_answer)
        model_key = self._normalize_model(model)
        created_at = dt.datetime.now(dt.timezone.utc)

        files_clean: List[str] = []
        if files:
            seen = set()
            for item in files:
                if isinstance(item, str) and item and item not in seen:
                    seen.add(item)
                    files_clean.append(item)

        embedding_blob = self._serialize_embedding(embedding)

        values = {
            "question": question,
            "question_normalized": norm_question,
            "answer": answer,
            "answer_normalized": norm_answer,
            "files": files_clean,
            "model": model,
            "model_key": model_key,
            "question_hash": question_hash,
            "answer_hash": answer_hash,
            "embedding": embedding_blob,
            "created_at": created_at,
        }

        stmt = pg_insert(self._table).values(values)
        excluded = stmt.excluded
        stmt = stmt.on_conflict_do_update(
            index_elements=[self._table.c.question_hash, self._table.c.model_key],
            set_={
                "question": excluded.question,
                "question_normalized": excluded.question_normalized,
                "answer": excluded.answer,
                "answer_normalized": excluded.answer_normalized,
                "files": excluded.files,
                "model": excluded.model,
                "answer_hash": excluded.answer_hash,
                "embedding": func.coalesce(excluded.embedding, self._table.c.embedding),
                "created_at": excluded.created_at,
            },
        )

        with self._lock:
            session = self._Session()
            try:
                session.execute(stmt)
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    def get_exact(self, normalized_question: str, model: Optional[str] = None) -> Optional[CacheHit]:
        if normalized_question is None:
            return None
        question_hash = self.hash_text(normalized_question)
        model_key = self._normalize_model(model)

        columns = [
            self._table.c.question,
            self._table.c.answer,
            self._table.c.files,
            self._table.c.model,
            self._table.c.question_hash,
            self._table.c.answer_hash,
            self._table.c.created_at,
        ]

        def _fetch(session, key: str) -> Optional[Any]:
            stmt = (
                select(*columns)
                .where(self._table.c.question_hash == question_hash)
                .where(self._table.c.model_key == key)
                .order_by(self._table.c.created_at.desc())
                .limit(1)
            )
            return session.execute(stmt).first()

        with self._lock:
            session = self._Session()
            try:
                row = _fetch(session, model_key)
                if row is None and model_key:
                    row = _fetch(session, "")
            finally:
                session.close()

        if row is None:
            return None

        mapping = row._mapping
        files = self._coerce_files(mapping["files"])
        created_at = mapping["created_at"]
        created_ts = created_at.timestamp() if hasattr(created_at, "timestamp") else float(created_at)

        return CacheHit(
            question=mapping["question"],
            answer=mapping["answer"],
            files=files,
            model=mapping["model"],
            question_hash=mapping["question_hash"],
            answer_hash=mapping["answer_hash"],
            created_at=created_ts,
            similarity=1.0,
        )

    def find_similar(
        self,
        embedding: Sequence[float],
        model: Optional[str] = None,
        threshold: float = 0.92,
    ) -> Optional[CacheHit]:
        vector = list(embedding or [])
        if not vector:
            return None

        model_key = self._normalize_model(model)
        keys = [model_key] if model_key else [""]
        if model_key:
            keys.append("")

        stmt = (
            select(
                self._table.c.question,
                self._table.c.answer,
                self._table.c.files,
                self._table.c.model,
                self._table.c.question_hash,
                self._table.c.answer_hash,
                self._table.c.created_at,
                self._table.c.embedding,
            )
            .where(self._table.c.embedding.isnot(None))
            .where(self._table.c.model_key.in_(keys))
            .order_by(self._table.c.created_at.desc())
            .limit(self.similarity_scan_limit)
        )

        with self._lock:
            session = self._Session()
            try:
                rows = session.execute(stmt).all()
            finally:
                session.close()

        best_hit: Optional[CacheHit] = None
        best_score = threshold
        for row in rows:
            mapping = row._mapping
            stored_vec = self._deserialize_embedding(mapping["embedding"])
            if not stored_vec or len(stored_vec) != len(vector):
                continue
            score = self._cosine_similarity(vector, stored_vec)
            if score < threshold or score <= best_score:
                continue

            files = self._coerce_files(mapping["files"])
            created_at = mapping["created_at"]
            created_ts = created_at.timestamp() if hasattr(created_at, "timestamp") else float(created_at)

            best_hit = CacheHit(
                question=mapping["question"],
                answer=mapping["answer"],
                files=files,
                model=mapping["model"],
                question_hash=mapping["question_hash"],
                answer_hash=mapping["answer_hash"],
                created_at=created_ts,
                similarity=score,
            )
            best_score = score
        return best_hit

    def close(self) -> None:
        with self._lock:
            try:
                self._engine.dispose()
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
