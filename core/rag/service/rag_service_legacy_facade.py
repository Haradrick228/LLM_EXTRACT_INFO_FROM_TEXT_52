"""RAG wrapper for bots and scraper (logging + list normalization).

Post-answer LLM validation (editor): see archive/core_rag/rag_service_wrapper_with_validation.py.
"""
import os
import re
import logging

from core.db_service import DBService
from core.rag.service.rag_service import RagService
from core.services.qa_service import QAService

log = logging.getLogger("rag_wrapper")

class RagServiceWrapper:
    def __init__(self, settings, db_service=None):
        self.rag = RagService(
            client_id=settings.client_id,
            client_secret=settings.client_secret,
            folder_path=settings.folder_path,
            persist_directory=settings.persist_directory,
            scope=settings.scope,
            model=settings.model_name,
        )
        if db_service is None:
            db_service = DBService()
        self.qa_service = QAService(db_service)
        self.validator_llm = None
        try:
            from core.rag.llm.deepseek_llm_client import DeepSeekLLM
            if settings.openai_api_key:
                self.validator_llm = DeepSeekLLM(api_key=settings.openai_api_key, temperature=0.0)
        except Exception:
            pass

    def _normalize_numbered_list(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r':\s*(?=\d+[.)]\s)', ':\n', text)
        text = re.sub(r'(?<!\n)\s(?=\d+[.)]\s)', '\n', text)
        text = re.sub(r'(\d+)\.\s', r'\1) ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def answer(self, question: str, model: str = None):
        # Use model from parameter, or default to None (let RagService use its default)
        draft_answer = self.rag.answer_question(question, model_override=model)
        files = []

        try:
            preview = (draft_answer or "").strip().replace("\r", " ").replace("\n", " ")
            if len(preview) > 600:
                preview = preview[:600] + "…"
            log.info("[DRAFT] %s", preview)
        except Exception:
            pass

        final_answer = self._normalize_numbered_list(draft_answer)

        try:
            self.qa_service.log_question(0, question, model or "unknown")
            self.qa_service.log_answer(0, final_answer)
        except Exception:
            pass

        return final_answer, files

    def list_files(self) -> list[str]:
        try:
            collection = self.rag._vec.col.get(include=["metadatas"], limit=100000)
            files = [md.get("source") for md in collection.get("metadatas", []) if md and md.get("source")]
            unique = sorted({f for f in files if f and os.path.isfile(f)})
            return unique
        except Exception:
            return []
