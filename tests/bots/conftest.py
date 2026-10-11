"""Shared fixtures for Telegram bot handler tests."""
from unittest.mock import MagicMock, AsyncMock

import pytest
from aiogram import Bot

from common.config import Settings


@pytest.fixture
def settings():
    return Settings(
        admin_ids=[123],
        admin_usernames=["admin"],
    )


@pytest.fixture
def mock_bot():
    from unittest.mock import AsyncMock, MagicMock
    # Bot is awaited directly: await bot(SomeMethod(...))
    # So the bot itself must be an AsyncMock
    bot = AsyncMock()
    bot.id = 1
    # Return a message-like object when send_message is called
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=1))
    bot.delete_message = AsyncMock()
    bot.edit_message_text = AsyncMock()
    bot.send_document = AsyncMock()
    bot.answer_callback_query = AsyncMock()
    bot.get_updates = AsyncMock()
    return bot


@pytest.fixture
def mock_rag():
    wrapper = MagicMock()
    wrapper.answer.return_value = ("Тестовый ответ от RAG", [])
    wrapper.list_files.return_value = []
    return wrapper


@pytest.fixture
def mock_qa():
    qa = MagicMock()
    qa.save_feedback.return_value = True
    qa.get_question_count.return_value = 0
    qa.has_user_nps.return_value = False
    qa.save_nps.return_value = True
    return qa


@pytest.fixture
def mock_db():
    db = MagicMock()
    db.increment_question_count.return_value = None
    return db


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.send_message = AsyncMock()
    adapter.send_file = AsyncMock()
    adapter.download_file = AsyncMock(return_value="/tmp/test.pdf")
    return adapter


@pytest.fixture
def mock_file_library():
    fl = MagicMock()
    fl.get_page.return_value = (["file1.pdf", "file2.pdf"], None, None)
    return fl


@pytest.fixture
def mock_link_service():
    ls = MagicMock()
    ls.add_link = AsyncMock(return_value=(True, "OK"))
    ls.list_links.return_value = []
    return ls


@pytest.fixture
def mock_link_library():
    ll = MagicMock()
    ll.get_page.return_value = ([], None, None)
    return ll


@pytest.fixture
def mock_upload_service():
    us = MagicMock()
    us.add_file.return_value = (True, "OK")
    us.upload_folder = "/tmp/uploads"
    return us


def make_update_from_dict(raw: dict):
    """Build a proper aiogram Update from a raw dict."""
    from aiogram.types import Update
    return Update(**raw)


def make_message_dict(text: str, user_id: int = 1, chat_id: int = 1, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "message": {
            "message_id": 1000 + update_id,
            "from": {"id": user_id, "is_bot": False, "first_name": "Test", "username": "test_user"},
            "chat": {"id": chat_id, "type": "private", "username": "test_user"},
            "text": text,
            "date": 1700000000,
        }
    }


def make_callback_dict(data: str, user_id: int = 1, chat_id: int = 1, update_id: int = 1) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {
            "id": f"cb_{update_id}",
            "from": {"id": user_id, "is_bot": False, "first_name": "Test", "username": "test_user"},
            "message": {
                "message_id": 999,
                "chat": {"id": chat_id, "type": "private", "username": "test_user"},
                "text": "msg",
                "date": 1700000000,
            },
            "data": data,
            "chat_instance": "test_chat_instance",
        }
    }
