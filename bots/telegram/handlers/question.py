from __future__ import annotations

import asyncio
from typing import Any

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bots.telegram.handlers.states import MenuState
from bots.telegram.texts import (
    ASK_QUESTION_BUTTON,
    BACK_BUTTON,
    DEEPSEEK_BUTTON,
    GIGACHAT_BUTTON,
    GIGACHAT_PRO_BUTTON,
)
from bots.telegram.utils import is_admin
from common.config import Settings
from common.interfaces.user_interaction import UserInteraction

MODEL_OPTIONS = {
    GIGACHAT_BUTTON: "gigachat",
    DEEPSEEK_BUTTON: "deepseek",
    GIGACHAT_PRO_BUTTON: "gigachat_pro",
}
DEFAULT_MODEL = MODEL_OPTIONS[DEEPSEEK_BUTTON]
CHUNK_LIMIT = 4000
FEEDBACK_BUTTON_LABEL = "Leave feedback"
MODEL_PROMPT = "Choose a model for the answer:"
QUESTION_PROMPT = "Send your question as plain text."
FILES_AVAILABLE_PROMPT = "Documents retrieved from the knowledge base:"
FEEDBACK_PROMPT = "Did you get a helpful answer?"
MODEL_UNKNOWN_PROMPT = "The option was not recognised. Please pick one of the buttons."
EMPTY_QUESTION_PROMPT = "Please send a text question."
NO_ANSWER_FALLBACK = "No answer was generated."


class QuestionHandler:
    def __init__(self, adapter: UserInteraction, rag_service, db_service, settings: Settings) -> None:
        self.adapter = adapter
        self.rag_service = rag_service
        self.db = db_service
        self.settings = settings
        self.router = Router(name="question")
        self._register_routes()

    def _register_routes(self) -> None:
        @self.router.message(F.text == ASK_QUESTION_BUTTON)
        async def ask_question_prompt(message, state: FSMContext):
            if is_admin(self.settings, message.from_user.id, message.from_user.username):
                await state.set_state(MenuState.choose_model)
                await self.adapter.send_message(
                    user_id=message.from_user.id,
                    text=MODEL_PROMPT,
                    reply_markup=_build_model_keyboard(),
                )
                return

            await state.update_data(model=DEFAULT_MODEL)
            await state.set_state(MenuState.question)
            await _prompt_question(self.adapter, message.from_user.id)

        @self.router.message(StateFilter(MenuState.choose_model))
        async def handle_model_selection(message, state: FSMContext):
            model = MODEL_OPTIONS.get(message.text)
            if not model:
                await self.adapter.send_message(
                    user_id=message.from_user.id,
                    text=MODEL_UNKNOWN_PROMPT,
                )
                return

            await state.update_data(model=model)
            await state.set_state(MenuState.question)
            await _prompt_question(self.adapter, message.from_user.id)

        @self.router.message(StateFilter(MenuState.question))
        async def handle_question(message, state: FSMContext):
            user_id = message.from_user.id
            question_text = (message.text or "").strip()
            if not question_text:
                await self.adapter.send_message(user_id=user_id, text=EMPTY_QUESTION_PROMPT)
                return

            self.db.increment_question_count(user_id)
            await message.chat.do("typing")

            data = await state.get_data()
            model_name = data.get("model", DEFAULT_MODEL)

            answer_text, files = await asyncio.to_thread(
                self._ask_rag,
                question_text,
                model_name,
            )

            for chunk in _split_answer(answer_text):
                await self.adapter.send_message(user_id=user_id, text=chunk)

            if files:
                await self.adapter.send_message(user_id=user_id, text=FILES_AVAILABLE_PROMPT)
                for path in files:
                    await self.adapter.send_file(user_id=user_id, file_path=path)

            await state.update_data(
                question_text=question_text,
                answer_text=answer_text,
                model_name=model_name,
            )

            feedback_keyboard = InlineKeyboardBuilder()
            feedback_keyboard.button(text=FEEDBACK_BUTTON_LABEL, callback_data="feedback")
            await self.adapter.send_message(
                user_id=user_id,
                text=FEEDBACK_PROMPT,
                reply_markup=feedback_keyboard.as_markup(),
            )

    def _ask_rag(self, question: str, model_name: str) -> tuple[str, list[str]]:
        answer, files = self.rag_service.answer(question, model_name)
        return answer or "", files or []


def _build_model_keyboard() -> ReplyKeyboardMarkup:
    rows = [
        [KeyboardButton(text=GIGACHAT_BUTTON), KeyboardButton(text=DEEPSEEK_BUTTON)],
        [KeyboardButton(text=GIGACHAT_PRO_BUTTON), KeyboardButton(text=BACK_BUTTON)],
    ]
    return ReplyKeyboardMarkup(keyboard=rows, resize_keyboard=True)


def _prompt_question(adapter: UserInteraction, user_id: int) -> Any:
    keyboard = ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=BACK_BUTTON)]],
        resize_keyboard=True,
    )
    return adapter.send_message(user_id=user_id, text=QUESTION_PROMPT, reply_markup=keyboard)


def _split_answer(answer: str) -> list[str]:
    if not answer:
        return [NO_ANSWER_FALLBACK]
    return [
        answer[i : i + CHUNK_LIMIT]
        for i in range(0, len(answer), CHUNK_LIMIT)
    ]
