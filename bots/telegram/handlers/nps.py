from aiogram import Router, F
from aiogram.types import CallbackQuery, Message, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, \
    KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import StateFilter

from bots.telegram.handlers.states import MenuState

class NPSHandler:
    def __init__(self, adapter, db_service):
        self.adapter = adapter
        self.db = db_service
        self.router = Router()
        self._register_routes()

    def _register_routes(self):
        # 1) Получаем балл NPS
        @self.router.callback_query(
            StateFilter(MenuState.waiting_for_nps_rating),
            F.data.startswith("nps:")
        )
        async def receive_nps(call: CallbackQuery, state: FSMContext):
            await call.answer()
            score = int(call.data.split(":", 1)[1])
            await state.update_data(nps_score=score)
            await state.set_state(MenuState.waiting_for_nps_comment)

            kb = InlineKeyboardMarkup(
                inline_keyboard=[[
                    InlineKeyboardButton(text="🔙 Пропустить", callback_data="nps_skip_comment")
                ]]
            )
            await call.message.answer(
                "💬 Вы можете оставить комментарий к оценке (или нажмите «🔙 Пропустить»):",
                reply_markup=kb,
            )

        # 2) Получаем комментарий
        @self.router.message(StateFilter(MenuState.waiting_for_nps_comment))
        async def receive_nps_comment(message: Message, state: FSMContext):
            data = await state.get_data()
            score = data["nps_score"]
            comment = message.text if message.text != "🔙 Пропустить" else None

            self.db.save_nps(
                user_id=message.from_user.id,
                score=score,
                comment=comment
            )

            await message.reply("🙏 Спасибо за оценку работы бота!")

            # Приглашаем задать новый вопрос
            kb = ReplyKeyboardMarkup(
                keyboard=[[KeyboardButton(text="🔙 Назад")]],
                resize_keyboard=True,
            )
            await message.answer(
                "Если у вас есть ещё вопрос — просто напишите его.",
                reply_markup=kb,
            )
            await state.set_state(MenuState.question)

        # 3) Пропуск комментария
        @self.router.callback_query(
            StateFilter(MenuState.waiting_for_nps_comment),
            F.data == "nps_skip_comment"
        )
        async def skip_comment(call: CallbackQuery, state: FSMContext):
            await call.answer()
            data = await state.get_data()
            score = data["nps_score"]

            self.db.save_nps(
                user_id=call.from_user.id,
                score=score,
                comment=None
            )

            await call.message.answer("🙏 Спасибо за оценку работы бота!")

            # Приглашаем задать новый вопрос
            kb = ReplyKeyboardMarkup(
                keyboard=[[KeyboardButton(text="🔙 Назад")]],
                resize_keyboard=True,
            )
            await call.message.answer(
                "Если у вас есть ещё вопрос — просто напишите его.",
                reply_markup=kb,
            )
            await state.set_state(MenuState.question)