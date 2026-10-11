from aiogram import Router, F
from aiogram.types import CallbackQuery, Message, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import StateFilter
from bots.telegram.handlers.states import MenuState
from bots.telegram.texts import (
    CANNOT_RATE_BUTTON,
    FEEDBACK_COMMENT_PROMPT,
    FEEDBACK_NEXT_QUESTION_PROMPT,
    FEEDBACK_NPS_PROMPT,
    FEEDBACK_RATING_PROMPT,
    FEEDBACK_THANKS_NO_COMMENT,
    FEEDBACK_THANKS_WITH_COMMENT,
    SKIP_BUTTON,
    BACK_BUTTON,
)

class FeedbackHandler:
    """Собирает обратную связь: оценка ответа (1–10), комментарий, NPS.

    После 8 вопросов предлагает пользователю пройти опрос NPS
    (Net Promoter Score) с оценкой от 0 до 10.
    """
    NPS_THRESHOLD = 8  # после этого числа вопросов предлагается NPS

    def __init__(self, adapter, db_service):
        self.adapter = adapter
        self.db = db_service
        self.router = Router()
        self._register_routes()

    def _register_routes(self):
        # 1) Нажатие “📝 Оценить ответ”
        @self.router.callback_query(StateFilter(MenuState.question), F.data == "feedback")
        async def prompt_rating(call: CallbackQuery, state: FSMContext):
            await call.answer()
            await call.message.delete()

            kb = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text=str(i), callback_data=f"rating:{i}") for i in range(1, 6)],
                    [InlineKeyboardButton(text=str(i), callback_data=f"rating:{i}") for i in range(6, 11)],
                    [InlineKeyboardButton(text=CANNOT_RATE_BUTTON, callback_data="rating:na")],
                ]
            )
            await call.message.answer(
                FEEDBACK_RATING_PROMPT,
                reply_markup=kb,
            )
            await state.set_state(MenuState.waiting_for_rating)

        # 2) Пользователь выбрал рейтинг (1–10 или na)
        @self.router.callback_query(StateFilter(MenuState.waiting_for_rating), F.data.startswith("rating:"))
        async def handle_rating(call: CallbackQuery, state: FSMContext):
            await call.answer()
            await call.message.delete()

            _, payload = call.data.split(":", 1)
            rating = int(payload) if payload.isdigit() else 0
            await state.update_data(rating=rating)

            # Просим комментарий или “Пропустить”
            kb2 = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text=SKIP_BUTTON, callback_data="skip_comment")]
                ]
            )
            prompt = await call.message.answer(
                FEEDBACK_COMMENT_PROMPT,
                reply_markup=kb2,
            )
            await state.update_data(comment_prompt_id=prompt.message_id)
            await state.set_state(MenuState.waiting_for_comment)

        # 3) Пользователь ввёл текстовый комментарий
        @self.router.message(StateFilter(MenuState.waiting_for_comment))
        async def receive_comment(message: Message, state: FSMContext):
            uid = message.from_user.id
            data = await state.get_data()

            # удаляем просьбу и сообщение-комментарий
            prompt_id = data.get("comment_prompt_id")
            if prompt_id:
                await message.bot.delete_message(chat_id=uid, message_id=prompt_id)
            comment = message.text.strip()
            await message.delete()

            # Сохраняем фидбэк (рейтинг + комментарий)
            self.db.save_feedback(
                user_id=uid,
                question=data.get("question_text"),
                answer=data.get("answer_text"),
                model=data.get("model_name"),
                rating=data.get("rating", 0),
                comment=comment,
            )
            await message.answer(FEEDBACK_THANKS_WITH_COMMENT)

            # Проверяем порог для NPS
            count = self.db.get_question_count(uid)
            if count >= self.NPS_THRESHOLD and not self.db.has_user_nps(uid):
                # Строим клавиатуру NPS 0–10
                nps_kb = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [InlineKeyboardButton(text=str(i), callback_data=f"nps:{i}") for i in range(0, 6)],
                        [InlineKeyboardButton(text=str(i), callback_data=f"nps:{i}") for i in range(6, 11)],
                    ]
                )
                await message.answer(
                    FEEDBACK_NPS_PROMPT,
                    reply_markup=nps_kb,
                )
                await state.set_state(MenuState.waiting_for_nps_rating)
                return

            # Иначе: возвращаем приглашение к новому вопросу
            await message.answer(
                text=FEEDBACK_NEXT_QUESTION_PROMPT,
                reply_markup=ReplyKeyboardMarkup(
                    keyboard=[[KeyboardButton(text=BACK_BUTTON)]],
                    resize_keyboard=True,
                ),
            )
            await state.set_state(MenuState.question)

        # 4) Пользователь нажал “Пропустить” вместо комментария
        @self.router.callback_query(StateFilter(MenuState.waiting_for_comment), F.data == "skip_comment")
        async def skip_comment(call: CallbackQuery, state: FSMContext):
            uid = call.from_user.id
            data = await state.get_data()
            await call.answer()
            await call.message.delete()

            # Сохраняем фидбэк (рейтинг без комментария)
            self.db.save_feedback(
                user_id=uid,
                question=data.get("question_text"),
                answer=data.get("answer_text"),
                model=data.get("model_name"),
                rating=data.get("rating", 0),
                comment=None,
            )
            await call.message.answer(FEEDBACK_THANKS_NO_COMMENT)

            # Проверяем NPS
            count = self.db.get_question_count(uid)
            if count >= self.NPS_THRESHOLD and not self.db.has_user_nps(uid):
                nps_kb = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [InlineKeyboardButton(text=str(i), callback_data=f"nps:{i}") for i in range(0, 6)],
                        [InlineKeyboardButton(text=str(i), callback_data=f"nps:{i}") for i in range(6, 11)],
                    ]
                )
                await call.message.answer(
                    FEEDBACK_NPS_PROMPT,
                    reply_markup=nps_kb,
                )
                await state.set_state(MenuState.waiting_for_nps_rating)
                return

            await call.message.answer(
                text=FEEDBACK_NEXT_QUESTION_PROMPT,
                reply_markup=ReplyKeyboardMarkup(
                    keyboard=[[KeyboardButton(text=BACK_BUTTON)]],
                    resize_keyboard=True,
                ),
            )
            await state.set_state(MenuState.question)
