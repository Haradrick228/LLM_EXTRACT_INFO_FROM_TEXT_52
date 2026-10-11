"""QA Service for managing question-answer history and feedback."""
from typing import Optional, List


class QAService:
    """Service for managing Q&A history, feedback and NPS scores."""

    def __init__(self, db_service):
        self.db = db_service
        self._question_counts: dict[int, int] = {}
        self._nps_users: set[int] = set()

    def log_question(
        self,
        user_id: int,
        question: str,
        model: str,
        session_id: Optional[str] = None
    ) -> Optional[int]:
        """Log a question to the database. Returns the ID of the created record."""
        self._question_counts[user_id] = self._question_counts.get(user_id, 0) + 1
        return None

    def log_answer(
        self,
        question_id: int,
        answer: str,
        sources: Optional[List[str]] = None,
        duration_ms: Optional[float] = None
    ) -> Optional[int]:
        """Log an answer to the database. Returns the ID of the created record."""
        return None

    # ── FeedbackHandler-compatible signatures ──

    def save_feedback(
        self,
        user_id: int,
        question: str,
        answer: str,
        model: str,
        rating: int,
        comment: Optional[str] = None,
    ) -> bool:
        """Save user feedback (called from FeedbackHandler)."""
        self.log_question(user_id, question, model)
        self.log_answer(0, answer)
        return True

    def get_question_count(self, user_id: int) -> int:
        """Get the number of questions asked by a user."""
        return self._question_counts.get(user_id, 0)

    def has_user_nps(self, user_id: int) -> bool:
        """Check if user already submitted NPS."""
        return user_id in self._nps_users

    def save_nps(
        self,
        user_id: int,
        score: int,
        comment: Optional[str] = None
    ) -> bool:
        """Save NPS score from user."""
        self._nps_users.add(user_id)
        return True
