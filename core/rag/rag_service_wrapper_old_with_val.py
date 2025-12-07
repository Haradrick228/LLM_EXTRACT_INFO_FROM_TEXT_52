import os
import re
import logging

from core.db_service import DBService
from core.rag.rag_service import RagService
from core.services.qa_service import QAService
from core.rag.deepseek_llm import DeepSeekLLM
from core.services.validation_service import ValidationService

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
        self.validator_llm = DeepSeekLLM(api_key=settings.openai_api_key, temperature=0.0)
        self.validator = ValidationService(llm=self.validator_llm, temperature=0.0)

    def _normalize_numbered_list(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r":\s*(?=\d+[.)]\s)", ":\n", text)
        text = re.sub(r"(?<!\n)\s(?=\d+[.)]\s)", "\n", text)
        text = re.sub(r"(\d+)\.\s", r"\1) ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def answer(self, question: str, model: str = None):
        draft_answer = self.rag.answer(question, model=model)
        files = []

        try:
            preview = (draft_answer or "").strip().replace("\r", " ").replace("\n", " ")
            if len(preview) > 600:
                preview = preview[:600] + "?"
            log.info("[DRAFT] %s", preview)
        except Exception:
            pass

        final_answer = draft_answer
        try:
            res = self.validator.rewrite(question=question, draft_answer=draft_answer)
            final_answer = res.get("final_answer") or draft_answer
            if not (final_answer or "").strip():
                final_answer = draft_answer
            final_answer = self._normalize_numbered_list(final_answer)
            try:
                log.info(
                    "[EDITOR] clarity=%s brevity=%s flags=%s | len(draft)=%d ~ len(final)=%d",
                    res.get("clarity"),
                    res.get("brevity"),
                    ",".join(res.get("safety_flags") or []),
                    len(draft_answer or ""),
                    len(final_answer or ""),
                )
            except Exception:
                pass
        except Exception as e:
            log.exception("Editor failed: %s", e)
            final_answer = self._normalize_numbered_list(draft_answer)

        try:
            self.qa_service.log(question, final_answer)
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
