from aiogram import Router, F
from aiogram.types import CallbackQuery, Message, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import StateFilter
from bots.telegram.handlers.states import MenuState

class FeedbackHandler:
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
                    [InlineKeyboardButton(text="❓ Не могу оценить", callback_data="rating:na")],
                ]
            )
            await call.message.answer(
                "Пожалуйста, оцените достоверность ответа от 1 до 10:",
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
                    [InlineKeyboardButton(text="🔙 Пропустить", callback_data="skip_comment")]
                ]
            )
            prompt = await call.message.answer(
                "💬 Теперь напишите комментарий (или нажмите «Пропустить»):",
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
            await message.answer("🙏 Спасибо за ваш комментарий и отзыв!")

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
                    "📊 Спасибо, что пользуетесь ботом! "
                    "Оцените его по шкале от 0 до 10, насколько вы порекомендуете бота друзьям или коллегам:",
                    reply_markup=nps_kb,
                )
                await state.set_state(MenuState.waiting_for_nps_rating)
                return

            # Иначе: возвращаем приглашение к новому вопросу
            await message.answer(
                "Если у вас есть ещё вопрос — просто напишите его.",
                reply_markup=ReplyKeyboardMarkup(
                    keyboard=[[KeyboardButton(text="🔙 Назад")]],
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
            await call.message.answer("🙏 Спасибо за ваш отзыв!")

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
                    "📊 Спасибо, что пользуетесь ботом! "
                    "Оцените его по шкале от 0 до 10, насколько вы порекомендуете бота друзьям или коллегам:",
                    reply_markup=nps_kb,
                )
                await state.set_state(MenuState.waiting_for_nps_rating)
                return

            await call.message.answer(
                "Если у вас есть ещё вопрос — просто напишите его.",
                reply_markup=ReplyKeyboardMarkup(
                    keyboard=[[KeyboardButton(text="🔙 Назад")]],
                    resize_keyboard=True,
                ),
            )
            await state.set_state(MenuState.question)
