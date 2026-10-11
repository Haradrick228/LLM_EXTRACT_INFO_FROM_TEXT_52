from aiogram import Router, F
from aiogram.types import CallbackQuery, Message, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, \
    KeyboardButton
from aiogram.fsm.context import FSMContext
from aiogram.filters.state import StateFilter

from bots.telegram.handlers.states import MenuState
from bots.telegram.texts import BACK_BUTTON, FEEDBACK_NEXT_QUESTION_PROMPT, NPS_COMMENT_PROMPT, NPS_THANKS, SKIP_BUTTON

class NPSHandler:
    """Принимает оценку NPS (0–10) и опциональный комментарий."""
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
                    InlineKeyboardButton(text=SKIP_BUTTON, callback_data="nps_skip_comment")
                ]]
            )
            await call.message.answer(
                NPS_COMMENT_PROMPT,
                reply_markup=kb,
            )

        # 2) Получаем комментарий
        @self.router.message(StateFilter(MenuState.waiting_for_nps_comment))
        async def receive_nps_comment(message: Message, state: FSMContext):
            data = await state.get_data()
            score = data["nps_score"]
            comment = message.text if message.text != SKIP_BUTTON else None

            self.db.save_nps(
                user_id=message.from_user.id,
                score=score,
                comment=comment
            )

            await message.reply(NPS_THANKS)

            # Приглашаем задать новый вопрос
            kb = ReplyKeyboardMarkup(
                keyboard=[[KeyboardButton(text=BACK_BUTTON)]],
                resize_keyboard=True,
            )
            await message.answer(
                FEEDBACK_NEXT_QUESTION_PROMPT,
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

            await call.message.answer(NPS_THANKS)

            # Приглашаем задать новый вопрос
            kb = ReplyKeyboardMarkup(
                keyboard=[[KeyboardButton(text=BACK_BUTTON)]],
                resize_keyboard=True,
            )
            await call.message.answer(
                FEEDBACK_NEXT_QUESTION_PROMPT,
                reply_markup=kb,
            )
            await state.set_state(MenuState.question)