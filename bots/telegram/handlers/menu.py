"""Helpers for building Telegram reply keyboards."""
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

from common.config import Settings
from bots.telegram.texts import (
    ADD_RESOURCE_BUTTON,
    ASK_QUESTION_BUTTON,
    FILE_LIBRARY_BUTTON,
    LINK_LIBRARY_BUTTON,
    UPLOAD_FILES_BUTTON,
)


def get_main_menu(settings: Settings, user_id: int, username: str | None = None) -> ReplyKeyboardMarkup:
    """Build the main menu keyboard tailored for admins and regular users."""
    is_admin = (
        user_id in settings.admin_ids
        or (username and username in settings.admin_usernames)
    )

    keyboard = [
        [
            KeyboardButton(text=ASK_QUESTION_BUTTON),
            KeyboardButton(text=LINK_LIBRARY_BUTTON),
        ]
    ]

    if is_admin:
        keyboard.append(
            [
                KeyboardButton(text=FILE_LIBRARY_BUTTON),
                KeyboardButton(text=ADD_RESOURCE_BUTTON),
            ]
        )
        keyboard.append([
            KeyboardButton(text=UPLOAD_FILES_BUTTON),
        ])

    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        one_time_keyboard=False,
    )

