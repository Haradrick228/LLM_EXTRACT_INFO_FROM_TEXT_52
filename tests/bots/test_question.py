"""Tests for QuestionHandler."""
import pytest
from aiogram.fsm.storage.memory import MemoryStorage, StorageKey
from aiogram import Dispatcher

from bots.telegram.handlers.question import QuestionHandler
from bots.telegram.handlers.states import MenuState
from bots.telegram.texts import (
    ASK_QUESTION_BUTTON, EMPTY_QUESTION_PROMPT, MODEL_PROMPT,
    QUESTION_PROMPT, FEEDBACK_PROMPT, DEEPSEEK_BUTTON,
)


@pytest.fixture
def handler(mock_adapter, mock_rag, mock_db, settings):
    return QuestionHandler(mock_adapter, mock_rag, mock_db, settings)


def _upd(text, uid=1, mid=1):
    from conftest import make_update_from_dict, make_message_dict
    return make_update_from_dict(make_message_dict(text, uid, uid, mid))


def _key(bot_id=1, chat_id=1, user_id=1):
    return StorageKey(bot_id=bot_id, chat_id=chat_id, user_id=user_id)


@pytest.mark.asyncio
async def test_non_admin_skips_model_select(handler, mock_adapter, mock_bot):
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd(ASK_QUESTION_BUTTON, uid=999))
    text = mock_adapter.send_message.call_args.kwargs.get("text", "")
    assert QUESTION_PROMPT in text
    assert MODEL_PROMPT not in text


@pytest.mark.asyncio
async def test_admin_gets_model_select(handler, mock_adapter, mock_bot):
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd(ASK_QUESTION_BUTTON, uid=123))
    text = mock_adapter.send_message.call_args.kwargs.get("text", "")
    assert MODEL_PROMPT in text


@pytest.mark.asyncio
async def test_empty_question_rejected(handler, mock_adapter, mock_bot):
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(handler.router)
    await dp.feed_update(bot=mock_bot, update=_upd(ASK_QUESTION_BUTTON, uid=1, mid=1))
    await dp.feed_update(bot=mock_bot, update=_upd("   ", uid=1, mid=2))
    texts = []
    for call in mock_adapter.send_message.call_args_list:
        texts.append(call.kwargs.get("text", ""))
    assert any(EMPTY_QUESTION_PROMPT in t for t in texts)


@pytest.mark.asyncio
async def test_admin_full_flow(handler, mock_adapter, mock_rag, mock_bot):
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(handler.router)
    for i, text in enumerate([ASK_QUESTION_BUTTON, DEEPSEEK_BUTTON, "Что такое Хрестоматия?"]):
        await dp.feed_update(bot=mock_bot, update=_upd(text, uid=123, mid=i + 1))
    mock_rag.answer.assert_called()
    texts = []
    for call in mock_adapter.send_message.call_args_list:
        texts.append(call.kwargs.get("text", ""))
    assert any(FEEDBACK_PROMPT in t for t in texts)
