"""Tests for FeedbackHandler."""
import pytest
from aiogram.fsm.storage.memory import MemoryStorage, StorageKey
from aiogram import Dispatcher

from bots.telegram.handlers.feedback import FeedbackHandler
from bots.telegram.handlers.states import MenuState
from bots.telegram.texts import ASK_QUESTION_BUTTON


@pytest.fixture
def handler(mock_adapter, mock_qa):
    return FeedbackHandler(mock_adapter, mock_qa)


def _upd(text, uid=1, mid=1):
    from conftest import make_update_from_dict, make_message_dict
    return make_update_from_dict(make_message_dict(text, uid, uid, mid))


def _cb(data, uid=1, mid=1):
    from conftest import make_update_from_dict, make_callback_dict
    return make_update_from_dict(make_callback_dict(data, uid, uid, mid))


def _key(bot_id=1, chat_id=1, user_id=1):
    return StorageKey(bot_id=bot_id, chat_id=chat_id, user_id=user_id)


@pytest.mark.asyncio
async def test_feedback_prompt_shows_rating(handler, mock_adapter, mock_bot):
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(handler.router)
    # Set question state so StateFilter matches
    await storage.set_state(_key(), MenuState.question)
    await dp.feed_update(bot=mock_bot, update=_cb("feedback", mid=1))


@pytest.mark.asyncio
async def test_feedback_saves_with_comment(handler, mock_qa, mock_bot):
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(handler.router)
    k = _key()
    await storage.set_state(k, MenuState.question)
    await dp.feed_update(bot=mock_bot, update=_cb("feedback", mid=1))
    await dp.feed_update(bot=mock_bot, update=_cb("rating:8", mid=2))
    await dp.feed_update(bot=mock_bot, update=_upd("Отличный ответ!", mid=3))
    mock_qa.save_feedback.assert_called()
    assert mock_qa.save_feedback.call_args.kwargs.get("rating") == 8


@pytest.mark.asyncio
async def test_feedback_skip_comment(handler, mock_qa, mock_bot):
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(handler.router)
    k = _key()
    await storage.set_state(k, MenuState.question)
    await dp.feed_update(bot=mock_bot, update=_cb("feedback", mid=1))
    await dp.feed_update(bot=mock_bot, update=_cb("rating:5", mid=2))
    await dp.feed_update(bot=mock_bot, update=_cb("skip_comment", mid=3))
    mock_qa.save_feedback.assert_called()
    assert mock_qa.save_feedback.call_args.kwargs.get("rating") == 5
    assert mock_qa.save_feedback.call_args.kwargs.get("comment") is None


@pytest.mark.asyncio
async def test_feedback_cannot_rate(handler, mock_qa, mock_bot):
    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    dp.include_router(handler.router)
    k = _key()
    await storage.set_state(k, MenuState.question)
    await dp.feed_update(bot=mock_bot, update=_cb("feedback", mid=1))
    await dp.feed_update(bot=mock_bot, update=_cb("rating:na", mid=2))
    await dp.feed_update(bot=mock_bot, update=_cb("skip_comment", mid=3))
    mock_qa.save_feedback.assert_called()
    assert mock_qa.save_feedback.call_args.kwargs.get("rating") == 0
