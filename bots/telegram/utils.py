"""Вспомогательные утилиты, общие для обработчиков Telegram-бота."""
from __future__ import annotations

from typing import Optional

from common.config import Settings


def is_admin(settings: Settings, user_id: int, username: Optional[str]) -> bool:
    """Возвращает True, если пользователь присутствует в списке администраторов по id или имени."""
    if user_id in settings.admin_ids:
        return True
    if username:
        return username in settings.admin_usernames
    return False


