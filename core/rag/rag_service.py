from __future__ import annotations

import os
import time
import uuid
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import re


def _load_env_once() -> None:
    try:
        from dotenv import load_dotenv, find_dotenv
        path = find_dotenv(usecwd=True)
        if path:
            # Override .env settings with localhost for local GPU eval
            if os.getenv("CHROMA_HOST_OVERRIDE") == "1":
                load_dotenv(path, override=True)
                os.environ["CHROMA_HOST"] = "127.0.0.1"
                os.environ["CHROMA_PORT"] = "18000"
                os.environ["CHROMA_SERVER_HOST"] = "127.0.0.1"
                os.environ["CHROMA_SERVER_HTTP_PORT"] = "18000"
            else:
                load_dotenv(path, override=False)
            logging.getLogger("RAG").info(f"[env] loaded .env from {path}")
    except Exception:
        pass


_LOG_LEVEL = os.getenv("RAG_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("RAG")

_load_env_once()

# По умолчанию отключаем кэш ответов, чтобы не ловить устаревшие заглушки.
os.environ.setdefault("LLM_CACHE_ENABLED", "0")

from chromadb import HttpClient
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:
    SentenceTransformer = None
    CrossEncoder = None

try:
    from core.graphrag.llm_provider import LLMClient as _LLMClient
except Exception:
    _LLMClient = None

GraphRetriever = None
_graph_retriever_import_error: Optional[str] = None
try:
    from core.graphrag.graph_retriever import GraphRetriever
except Exception as e1:
    try:
        from graph_retriever import GraphRetriever
    except Exception as e2:
        _graph_retriever_import_error = f"{e1!r} / {e2!r}"
        GraphRetriever = None

from core.rag.query_cache import QueryCache

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None


def _b(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _i(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except Exception:
        return default


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name) or default)
    except Exception:
        return default


@dataclass
class ChromaConfig:
    mode: str = os.getenv("CHROMA_MODE", "rest")
    host: str = os.getenv("CHROMA_HOST", "127.0.0.1")
    port: int = _i("CHROMA_PORT", 8000)
    tenant: str = os.getenv("CHROMA_TENANT", "default_tenant")
    database: str = os.getenv("CHROMA_DATABASE", "default_database")
    collection: str = os.getenv("CHROMA_COLLECTION", "default")
    persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma")
    embed_model: str = os.getenv("CHROMA_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")


@dataclass
class DBConfig:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = _i("POSTGRES_PORT", 5433)
    dbname: str = os.getenv("POSTGRES_DB", "")
    user: str = os.getenv("POSTGRES_USER", "")
    password: str = os.getenv("POSTGRES_PASSWORD", "")

    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.dbname} "
            f"user={self.user} password={self.password}"
        )


@dataclass
class GraphRAGConfig:
    enabled: bool = _b("GRAPH_RAG_ENABLED", True)
    level: int = _i("GRAPH_RAG_LEVEL", 1)
    max_anchors: int = _i("GRAPH_RAG_MAX_ANCHORS", 15)
    max_communities: int = _i("GRAPH_RAG_MAX_COMMUNITIES", 3)
    max_edges: int = _i("GRAPH_RAG_MAX_EDGES", 40)
    min_edge_score: float = _f("GRAPH_RAG_MIN_SCORE", 0.8)
    include_cites: bool = _b("GRAPH_RAG_INCLUDE_CITES", True)
    log_prompt_context: bool = _b("RAG_LOG_PROMPT_CONTEXT", True)
    max_prompt_ctx_chars: int = _i("RAG_MAX_PROMPT_CONTEXT_CHARS", 8000)


@dataclass
class LLMConfig:
    model: str = os.getenv("LLM_PROVIDER_MODEL", "deepseek-chat")
    temperature: float = _f("LLM_TEMPERATURE", 0.2)
    max_tokens: int = _i("LLM_MAX_TOKENS", 900)


class _SafeLLM:
    def __init__(self, cfg: LLMConfig):
        if _LLMClient is None:
            raise RuntimeError("LLMClient provider is not available (core.graphrag.llm_provider).")
        self.cfg = cfg
        try:
            self.client = _LLMClient(model=cfg.model)
        except TypeError:
            self.client = _LLMClient()

    def generate(self, prompt: str) -> str:
        t0 = time.time()
        if hasattr(self.client, "chat"):
            try:
                out = self.client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                )
                txt = (out or "").strip()
                log.info(f"[LLM] chat ok in {int((time.time()-t0)*1000)} ms, len={len(txt)}")
                return txt
            except Exception as e:
                log.warning(f"[LLM] chat failed: {e}; fallback to complete()")

        if hasattr(self.client, "complete"):
            out = self.client.complete(
                prompt=prompt,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            txt = (out or "").strip()
            log.info(f"[LLM] complete ok in {int((time.time()-t0)*1000)} ms, len={len(txt)}")
            return txt

        raise RuntimeError("LLMClient has neither chat(...) nor complete(...).")


class _VectorFallback:
    def __init__(self, cfg: ChromaConfig):
        self.cfg = cfg
        self.client = self._connect()
        self.col = self._open_collection()

    def _connect(self):
        if self.cfg.mode.lower() == "rest":
            # Clear and override env vars to prevent chromadb Settings from reading wrong values
            for key in list(os.environ.keys()):
                if key.startswith("CHROMA_"):
                    del os.environ[key]
            os.environ["CHROMA_SERVER_HOST"] = self.cfg.host
            os.environ["CHROMA_SERVER_HTTP_PORT"] = str(self.cfg.port)

            return HttpClient(
                host=self.cfg.host,
                port=self.cfg.port,
                settings=Settings(allow_reset=False, anonymized_telemetry=False),
                tenant=self.cfg.tenant,
                database=self.cfg.database,
            )
        raise RuntimeError("PersistentClient mode не поддерживается в этой сборке.")

    def _open_collection(self):
        name = self.cfg.collection
        embed_fn = SentenceTransformerEmbeddingFunction(model_name=self.cfg.embed_model)
        try:
            return self.client.get_collection(name, embedding_function=embed_fn)
        except Exception:
            log.warning(f"[Chroma] Collection '{name}' not found, creating:")
            return self.client.create_collection(name, embedding_function=embed_fn)

    def query(self, q: str, n: int = 5) -> List[Dict[str, Any]]:
        try:
            res = self.col.query(
                query_texts=[q],
                n_results=n,
                include=["metadatas", "documents", "distances", "embeddings"],
            )
            out: List[Dict[str, Any]] = []
            docs = (res or {}).get("documents", [[]])[0] if res else []
            metas = (res or {}).get("metadatas", [[]])[0] if res else []
            dists = (res or {}).get("distances", [[]])[0] if res else []
            seen = set()
            for i, doc in enumerate(docs):
                meta = metas[i] if i < len(metas) else {}
                dist = dists[i] if i < len(dists) else None
                sig = (meta.get("source"), meta.get("page_label"), meta.get("page"), hash(doc))
                if sig in seen:
                    continue
                seen.add(sig)
                out.append({"text": doc, "metadata": meta, "distance": dist})
            return out
        except Exception as e:
            msg = str(e)
            if "dimension" in msg.lower():
                log.error(f"[Chroma] query failed (embedding dim mismatch): {msg}")
            else:
                log.error(f"[Chroma] query failed: {msg}")
            return []


_GRAPH_PROMPT_TMPL = """ЭТАП 1. Представь, что ты помощник-аналитик благотворительного фонда. Отвечай только фактами из предоставленного графового контекста.

Контекст:
{graph_context}

Инструкция:
- Не выдумывай факты вне контекста.
- Если информации недостаточно, скажи об этом и предложи, что можно уточнить.
- Пиши кратко и по-деловому (по-русски).
Вопрос: {question}
Ответ:

ЭТАП 2 . Ты - редактор финального ответа для сайта благотворительного фонда <Вклад в будущее> Сбера. 
    "Работаешь ТОЛЬКО с ответом ответа и вопросом. "
    "Твоя цель - сделать ответ коротким, понятным и безопасным: без домыслов, без жаргона, без медсоветов. "
    "Нельзя добавлять новые факты, которых нет в черновике (никаких придуманных дат, сумм, условий, программ). "
    "Сохраняй цитаты из черновика в кавычках. Не используй Markdown и символ *."

Сделай следующее:
1) Убери повторы и <воду>, сохрани смысл. Если в черновике есть лишние обещания/обобщения (например, про <финансирование>), оставь только то, что явно сказано в черновике.
2) Безопасность и терминология:
   - Медицинские термины (<ДЦП>, <РАС>, <СДВГ>) - допустимы и нейтральны.
   - Не давай медицинских рекомендаций/диагнозов/лечения; если в вопросе просят медсовет - добавь короткое пояснение, что ты его не даёшь.
   - Используй язык <сначала человек> (напр., <ребёнок с ДЦП>).
3) Структура:
   - Если <что это/чем занимается> - 2 короткий абзац + при необходимости 3-10 пунктов.
   - Если <как участвовать/получить> - кто/для кого, шаги, где посмотреть сроки/контакты (только если они уже есть в черновике).
4) Ограничения: цель - ? 1000 символов; списков не более 7 пунктов, если они есть; без Markdown и символа *.
5) Формат списков: если есть пунткы, то каждый пункт с новой строки, нумерация вида "1) ...", "2) ...". Поставь пустую строку перед списком.

"""


def _chop(s: str, n: int) -> str:
    if n <= 0:
        return ""
    return s if len(s) <= n else s[: n - 1] + ":"


def _fmt_graph_context(ctx: Dict[str, Any], include_cites: bool, max_chars: int) -> str:
    parts: List[str] = []

    comm_summ = ctx.get("community_summaries") or []
    if comm_summ:
        block_lines: List[str] = []
        for i, s in enumerate(comm_summ, 1):
            txt = (s.get("summary") or "").strip()
            lvl = s.get("level")
            cid = s.get("community_id")
            head = f"[Сводка сообщества #{i} (L{lvl}, id={cid})]"
            if txt:
                block_lines.append(head + "\n" + _chop(txt, 1200))
        if block_lines:
            parts.append("\n\n".join(block_lines))

    structured = ctx.get("structured") or {}
    prog = structured.get("programs") or []
    part = structured.get("partners") or []
    aud = structured.get("audiences") or []
    if prog or part or aud:
        lines = ["[Структура]"]
        if prog:
            lines.append("• Программы/проекты: " + _chop(", ".join(prog[:12]), 600))
        if part:
            lines.append("• Партнёры: " + _chop(", ".join(part[:12]), 600))
        if aud:
            lines.append("• Аудитории/адресаты: " + _chop(", ".join(aud[:12]), 600))
        parts.append("\n".join(lines))

    edges = ctx.get("edges") or []
    if edges:
        lines = ["[Связи]"]
        for e in edges[: ctx.get("limit_edges", 40)]:
            head = e.get("head") or "?"
            rel = e.get("rel") or "related_to"
            tail = e.get("tail") or "?"
            score = e.get("score")
            cite = e.get("cite")
            row = f"- {head} -{rel}→ {tail}"
            if score is not None:
                try:
                    row += f" [score={float(score):.2f}]"
                except Exception:
                    pass
            if include_cites and cite:
                c = str(cite).strip()
                if c:
                    row += "\n   <" + _chop(c, 240) + ">"
            lines.append(row)
        parts.append("\n".join(lines))

    anchors = ctx.get("anchors") or []
    if anchors:
        a_fmt = ", ".join([f"{a.get('name','?')} ({a.get('type','?')})" for a in anchors])
        parts.append(f"[Якорные узлы] {a_fmt}")

    text = "\n\n".join(parts).strip()
    return _chop(text, max_chars)


_ALLOWED_PREDICATES_DEFAULT = {
    "includes", "supports", "partners_with", "offers", "used_for",
    "targets", "improves", "enables", "develops", "aligns_with",
    "author", "conducts", "regulates", "requires", "part_of", "is_a",
    "uses", "has", "based_on", "developed_by"
}

_PRED_RE = re.compile(r"-\s*([a-zA-Z_]+)\s*→")
_SCORE_RE = re.compile(r"\[score=([0-9]+(?:\.[0-9]+)?)\]")
_EDGE_ID_RE = re.compile(r"\[#edge:(\d+)\]")
_HEAD_REL_TAIL_RE = re.compile(r"-\s*(.*?)\s*-\s*([a-zA-Z_]+)\s*→\s*(.*?)(?:\s*\[|$)")


def _parse_edge_meta_from_str(edge_line: str):
    pred = None
    score = None
    edge_id = None
    try:
        m = _PRED_RE.search(edge_line)
        if m:
            pred = m.group(1).strip()
        m = _SCORE_RE.search(edge_line)
        if m:
            score = float(m.group(1))
        m = _EDGE_ID_RE.search(edge_line)
        if m:
            edge_id = m.group(1)
    except Exception:
        pass
    head = tail = None
    try:
        m2 = _HEAD_REL_TAIL_RE.search(edge_line)
        if m2:
            head = m2.group(1).strip() or None
            if not pred:
                pred = m2.group(2).strip()
            tail = m2.group(3).strip() or None
    except Exception:
        pass
    return head, pred, tail, score, edge_id


def _filter_graph_edges(
    edges_in: List[Any],
    min_score: float = 0.0,
    allowed_predicates: Optional[Set[str]] = None,
    max_edges: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not edges_in:
        return []

    allowed = set(p.lower() for p in (allowed_predicates or _ALLOWED_PREDICATES_DEFAULT))

    def _norm_pred(v: Optional[str]) -> str:
        return (v or "").strip().lower().replace(" ", "_")

    best_by_key: Dict[str, tuple[Optional[float], Dict[str, Any]]] = {}

    for e in edges_in:
        if isinstance(e, dict):
            head = e.get("head") or "?"
            rel = e.get("rel") or e.get("predicate") or "related_to"
            tail = e.get("tail") or "?"
            score = e.get("score")
            try:
                score = float(score) if score is not None else None
            except Exception:
                score = None
            pred_norm = _norm_pred(rel)
            if pred_norm and pred_norm not in allowed:
                continue
            if score is not None and score < min_score:
                continue
            eid = e.get("edge_id") or e.get("id") or f"{head}|{pred_norm}|{tail}"
            prev = best_by_key.get(eid)
            if not prev or (score is not None and (prev[0] is None or score > prev[0])):
                edge_obj = dict(e)
                edge_obj["rel"] = rel
                edge_obj["head"] = head
                edge_obj["tail"] = tail
                edge_obj["score"] = score
                edge_obj.setdefault("edge_id", eid)
                best_by_key[eid] = (score, edge_obj)
        else:
            s = str(e)
            head, rel, tail, score, eid = _parse_edge_meta_from_str(s)
            pred_norm = _norm_pred(rel)
            if pred_norm and pred_norm not in allowed:
                continue
            if score is not None and score < min_score:
                continue
            if not eid:
                eid = f"{head or '?'}|{pred_norm}|{tail or '?'}"
            edge_obj = {
                "head": head or "?",
                "rel": rel or "related_to",
                "tail": tail or "?",
                "score": score,
                "cite": s,
                "edge_id": eid,
            }
            prev = best_by_key.get(eid)
            if not prev or (score is not None and (prev[0] is None or score > prev[0])):
                best_by_key[eid] = (score, edge_obj)

    items = list(best_by_key.values())
    items.sort(key=lambda x: (x[0] is not None, x[0]), reverse=True)
    result = [edge for _, edge in items]
    if max_edges is not None and max_edges > 0:
        result = result[:max_edges]
    return result


class RagService:
    def __init__(self, **kwargs):
        model_override = kwargs.get("model")
        graph_enabled_override = kwargs.get("graph_enabled")
        chroma_host = kwargs.get("chroma_host")
        chroma_port = kwargs.get("chroma_port")
        chroma_collection = kwargs.get("chroma_collection")
        db_over = kwargs.get("db")

        self.db_cfg = DBConfig()
        self.chroma_cfg = ChromaConfig()
        self.graph_cfg = GraphRAGConfig()
        self.llm_cfg = LLMConfig()
        self.retriever_k = int(os.getenv("RETRIEVER_K", "5") or 5)
        self.retriever_fetch_k = int(os.getenv("RETRIEVER_FETCH_K", "20") or 20)
        self.rerank_model_name = os.getenv("RERANK_MODEL", self.chroma_cfg.embed_model)
        self.rerank_mode = os.getenv("RERANK_MODE", "auto").lower()

        if isinstance(model_override, str) and model_override:
            self.llm_cfg.model = model_override
        if graph_enabled_override is not None:
            self.graph_cfg.enabled = bool(graph_enabled_override)
        if isinstance(chroma_host, str) and chroma_host:
            self.chroma_cfg.host = chroma_host
        if isinstance(chroma_port, int):
            self.chroma_cfg.port = chroma_port
        if isinstance(chroma_collection, str) and chroma_collection:
            self.chroma_cfg.collection = chroma_collection
        if isinstance(db_over, dict) and db_over:
            self.db_cfg.host = db_over.get("host", self.db_cfg.host)
            self.db_cfg.port = int(db_over.get("port", self.db_cfg.port))
            self.db_cfg.dbname = db_over.get("dbname", self.db_cfg.dbname)
            self.db_cfg.user = db_over.get("user", self.db_cfg.user)
            self.db_cfg.password = db_over.get("password", self.db_cfg.password)

        self.perf_id: Optional[str] = None
        self.llm = _SafeLLM(self.llm_cfg)
        self._graph: Optional[Any] = None
        self._graph_ready: bool = False
        self._graph_fail_reason: Optional[str] = None
        self._vec = _VectorFallback(self.chroma_cfg)
        self._rerank_model: Optional[Any] = None
        self._rerank_type: str = "none"
        if SentenceTransformer is not None:
            try:
                wants_cross = "cross-encoder" in self.rerank_model_name or self.rerank_mode == "cross"
                if wants_cross and CrossEncoder is not None:
                    self._rerank_model = CrossEncoder(self.rerank_model_name)
                    self._rerank_type = "cross"
                else:
                    self._rerank_model = SentenceTransformer(self.rerank_model_name)
                    self._rerank_type = "bi"
                log.info("Rerank model: %s (%s)", self.rerank_model_name, self._rerank_type)
            except Exception as e:
                self._rerank_model = None
                self._rerank_type = "none"
                log.warning("Не удалось инициализировать модель ранжирования %s: %s", self.rerank_model_name, e)

        try:
            default_threshold = float(os.getenv("LLM_CACHE_SIMILARITY", "0.92"))
        except ValueError:
            default_threshold = 0.92
        self.cache_similarity_threshold = max(0.0, min(1.0, default_threshold))
        self.cache_scan_limit = max(1, int(os.getenv("LLM_CACHE_SIMILARITY_SCAN", "200")))
        self.cache = None
        self._query_embedder = self._build_query_embedder()
        self._cache_embed_failed = False
        cache_enabled = os.getenv("LLM_CACHE_ENABLED", "1") != "0"
        if cache_enabled:
            try:
                dsn = (os.getenv("LLM_CACHE_DSN") or "").strip()
                if not dsn:
                    dsn = QueryCache.resolve_dsn_from_env()
                schema_env = (os.getenv("LLM_CACHE_SCHEMA", "rag") or "").strip()
                table_env = (os.getenv("LLM_CACHE_TABLE", "cached_answers") or "").strip()
                if not dsn:
                    raise ValueError("Не задано подключение к БД для кэша (LLM_CACHE_DSN или POSTGRES_*).")
                schema = schema_env or None
                table = table_env or "cached_answers"
                self.cache = QueryCache(
                    dsn=dsn,
                    schema=schema,
                    table=table,
                    similarity_scan_limit=self.cache_scan_limit,
                )
                log.info("LLM cache enabled (PostgreSQL): %s.%s", schema or "public", table)
            except Exception as exc:
                self.cache = None
                log.warning("Не удалось инициализировать кэш ответов: %s", exc)
        else:
            log.info("LLM cache disabled (LLM_CACHE_ENABLED=0)")

        log.info(
            f"RagService инициализирован. GRAPH_RAG_ENABLED={self.graph_cfg.enabled} model={self.llm_cfg.model}"
        )

    def _perf_start(self, model: str, question: str):
        self.perf_id = uuid.uuid4().hex[:8]
        log.info(f"[perf:{self.perf_id}] start model={model}")
        log.info(f"Новый запрос пользователя: {question}")

    def _ensure_graph(self):
        if self._graph_ready or self._graph_fail_reason:
            return
        if not self.graph_cfg.enabled:
            self._graph_fail_reason = "disabled by env"
            return
        if GraphRetriever is None:
            self._graph_fail_reason = f"GraphRetriever import failed: {_graph_retriever_import_error}"
            return

        try:
            self._graph = GraphRetriever(
                level=self.graph_cfg.level,
                max_anchors=self.graph_cfg.max_anchors,
                max_communities=self.graph_cfg.max_communities,
                max_edges=self.graph_cfg.max_edges,
                min_edge_score=self.graph_cfg.min_edge_score,
            )
            self._graph_ready = True
            log.info("[GraphRAG] GraphRetriever готов.")
        except Exception as e:
            self._graph_fail_reason = f"init failed: {e!r}"
            log.warning(f"[GraphRAG] init failed: {e}")

    def _build_query_embedder(self) -> Optional[Any]:
        if HuggingFaceEmbeddings is None:
            log.warning("HuggingFaceEmbeddings недоступен, кэш похожих запросов работает только по точным совпадениям.")
            return None
        model_name = os.getenv("LLM_CACHE_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
        try:
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            log.warning("Не удалось инициализировать эмбеддер для кэша: %s", exc)
            return None

    def _embed_query_for_cache(self, text: str) -> Optional[List[float]]:
        if not text or self._query_embedder is None or self._cache_embed_failed:
            return None
        try:
            return self._query_embedder.embed_query(text)
        except Exception as exc:
            if not self._cache_embed_failed:
                self._cache_embed_failed = True
                log.warning("Не удалось построить эмбеддинг запроса для кэша: %s", exc)
            return None

    def answer(self, question: str, model: Optional[str] = None) -> str:
        normalized_for_cache = QueryCache.normalize_text(question)
        question_embedding: Optional[List[float]] = None
        collected_files: List[str] = []

        model_to_use = model or self.llm_cfg.model
        if model and model != self.llm_cfg.model:
            log.info(f"[LLM] per-call model override: {self.llm_cfg.model} → {model_to_use}")
            llm = _SafeLLM(LLMConfig(model=model_to_use,
                                     temperature=self.llm_cfg.temperature,
                                     max_tokens=self.llm_cfg.max_tokens))
        else:
            llm = self.llm

        self._perf_start(model_to_use, question)
        t_total = time.time()

        cache_hit = None
        if self.cache:
            try:
                cache_hit = self.cache.get_exact(normalized_for_cache, model_to_use)
                if cache_hit:
                    log.info(f"[perf:{self.perf_id}] cache-hit sim={cache_hit.similarity:.3f}")
                    return cache_hit.answer
                question_embedding = self._embed_query_for_cache(normalized_for_cache)
                if question_embedding:
                    threshold = self.cache_similarity_threshold if self.cache_similarity_threshold is not None else 0.92
                    cache_hit = self.cache.find_similar(
                        question_embedding,
                        model=model_to_use,
                        threshold=threshold,
                    )
                    if cache_hit:
                        log.info(f"[perf:{self.perf_id}] cache-hit sim={cache_hit.similarity:.3f}")
                        return cache_hit.answer
            except Exception as exc:
                log.warning("Ошибка при обращении к кэшу ответов: %s", exc)

        graph_ctx: Optional[Dict[str, Any]] = None
        self._ensure_graph()

        if self._graph_ready and self._graph is not None:
            t0 = time.time()
            try:
                if hasattr(self._graph, "retrieve_context"):
                    graph_ctx = self._graph.retrieve_context(question)
                elif hasattr(self._graph, "retrieve"):
                    graph_ctx = self._graph.retrieve(question)
                else:
                    log.warning("[GraphRAG] failed: GraphRetriever has no retrieve* method")
                    graph_ctx = None
            except Exception as e:
                log.warning(f"[GraphRAG] failed: {e}")
                graph_ctx = None

            dt = int((time.time() - t0) * 1000)
            edges_n = len(graph_ctx.get("edges", [])) if graph_ctx else 0
            comm_n = len(graph_ctx.get("communities", [])) if graph_ctx else 0
            log.info(f"[perf:{self.perf_id}] graph.ms={dt} edges={edges_n} communities={comm_n}")

        if graph_ctx and graph_ctx.get("edges"):
            allowed_preds = getattr(self.graph_cfg, "allowed_predicates", None)
            graph_ctx["edges"] = _filter_graph_edges(
                graph_ctx.get("edges") or [],
                min_score=self.graph_cfg.min_edge_score,
                allowed_predicates=allowed_preds,
                max_edges=self.graph_cfg.max_edges,
            )
            if not graph_ctx["edges"]:
                graph_ctx = None

        if graph_ctx and graph_ctx.get("edges"):
            ctx_txt = _fmt_graph_context(
                graph_ctx,
                include_cites=self.graph_cfg.include_cites,
                max_chars=self.graph_cfg.max_prompt_ctx_chars,
            )
            if self.graph_cfg.log_prompt_context:
                log.info("[GraphRAG] prompt.context >>>\n" + ctx_txt)

            prompt = _GRAPH_PROMPT_TMPL.format(
                graph_context=ctx_txt,
                question=question,
            )
        else:
            if not self._graph_ready:
                log.warning("[GraphRAG] контекст не найден (graph init failed/disabled) → фолбэк на векторный RAG.")
            else:
                log.warning("[GraphRAG] контекст не найден → фолбэк на векторный RAG.")

            t0 = time.time()
            docs = self._vec.query(question, n=self.retriever_fetch_k)
            docs = self._rerank_docs(question, docs, top_k=self.retriever_k)
            dt = int((time.time() - t0) * 1000)
            log.info(f"[perf:{self.perf_id}] chroma.ms={dt} n={len(docs)}")

            if not docs:
                draft = "?? В базе знаний пока нет контекста для ответа. Попробуйте переформулировать вопрос."
                log.info(f"[DRAFT] {draft}")
                log.info(f"[perf:{self.perf_id}] done total_ms={int((time.time()-t_total)*1000)}")
                return draft

            ctx_lines: List[str] = []
            for d in docs:
                text = (d.get("text") or "").strip()
                if len(text) > 1200:
                    text = text[:1199] + ":"
                meta = d.get("metadata") or {}
                key = meta.get("doc_key") or meta.get("source") or "unknown"
                src = meta.get("source")
                if isinstance(src, str) and os.path.isfile(src) and src not in collected_files:
                    collected_files.append(src)
                ctx_lines.append(f"[{key}] {text}")
            fallback_ctx = "\n\n".join(ctx_lines)

            prompt = (
                "Ответь кратко и по делу, используя только факты из контекста ниже. "
                "Если недостаточно данных - скажи об этом.\n\n"
                f"Контекст:\n{fallback_ctx}\n\n"
                f"Вопрос: {question}\nОтвет:"
            )

        log.info("Генерация ответа через LLM:")
        t0 = time.time()
        try:
            answer = llm.generate(prompt)
        except Exception as e:
            log.error(f"[LLM] failed: {e}")
            answer = "?? Внутренняя ошибка при генерации ответа. Попробуйте ещё раз."
        dt = int((time.time() - t0) * 1000)

        if self.cache and answer:
            try:
                if question_embedding is None:
                    question_embedding = self._embed_query_for_cache(normalized_for_cache)
                normalized_answer = QueryCache.normalize_text(answer)
                self.cache.store(
                    question=question,
                    answer=answer,
                    files=collected_files,
                    model=model_to_use,
                    normalized_question=normalized_for_cache,
                    normalized_answer=normalized_answer,
                    embedding=question_embedding,
                )
            except Exception as exc:
                log.warning("Не удалось сохранить ответ в кэше: %s", exc)

        log.info(f"[perf:{self.perf_id}] llm.ms={dt} answer_chars={len(answer)}")
        log.info(f"[perf:{self.perf_id}] done total_ms={int((time.time()-t_total)*1000)}")
        log.info(f"[DRAFT] {_chop(answer.replace(os.linesep,' '), 200)}")
        return answer

    def _rerank_docs(self, question: str, docs: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not docs or self._rerank_model is None:
            return docs[:top_k]
        texts = [d.get("text") or "" for d in docs]
        try:
            if self._rerank_type == "cross":
                pairs = [(question, t) for t in texts]
                scores = self._rerank_model.predict(pairs).tolist()
            else:
                q_emb = self._rerank_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
                d_emb = self._rerank_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                scores = (d_emb @ q_emb).tolist()
            scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [d for d, _ in scored[:top_k]]
        except Exception as e:
            log.warning("Ошибка ранжирования: %s", e)
            return docs[:top_k]

    def answer_question(self, question: str, model_override: Optional[str] = None) -> str:
        def _contains_cjk(text: str) -> bool:
            return bool(re.search(r"[\u4e00-\u9fff]", text))

        retries = 3
        last = ""
        q = question
        while retries > 0:
            try:
                last = self.answer(q, model_override)
                if not _contains_cjk(last):
                    return last
                retries -= 1
                log.info("Повтор генерации из-за CJK в ответе, осталось попыток: %s", retries)
                q = f"{question} Ответь на русском языке, без китайских символов."
            except Exception as e:
                retries -= 1
                log.warning("Повтор генерации из-за ошибки LLM (%s), осталось попыток: %s", e, retries)
                last = "?? Внутренняя ошибка при генерации ответа. Попробуйте ещё раз."
        return last


if __name__ == "__main__":
    svc = RagService()
    q = os.getenv("DEBUG_QUESTION", "Вклад в будущее - что это?")
    print(svc.answer(q))
