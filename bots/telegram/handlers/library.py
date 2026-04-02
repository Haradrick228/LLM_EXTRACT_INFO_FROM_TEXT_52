from __future__ import annotations

from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from bots.telegram.handlers.states import MenuState
from bots.telegram.texts import FILE_LIBRARY_BUTTON, FILE_LIBRARY_NEXT, FILE_LIBRARY_PREV


class LibraryHandler:
    def __init__(self, adapter, file_library, rag_service) -> None:
        self.adapter = adapter
        self.file_library = file_library
        self.rag_service = rag_service
        self.router = Router(name="file-library")
        self._register_routes()

    def _register_routes(self) -> None:
        @self.router.message(F.text == FILE_LIBRARY_BUTTON)
        async def show_library(message, state: FSMContext):
            await state.set_state(MenuState.library)
            names, prev_page, next_page = self.file_library.get_page(0)

            if not names:
                await self.adapter.send_message(
                    message.from_user.id,
                    "No documents are available yet.",
                )
                await state.clear()
                return

            await self._send_page(
                chat_id=message.from_user.id,
                names=names,
                prev_page=prev_page,
                next_page=next_page,
            )
            await self.adapter.send_message(
                message.from_user.id,
                "Reply with the document number to receive the file.",
            )
            await state.update_data(library=self.rag_service.list_files())

        @self.router.callback_query(
            F.data.startswith("lib_page:"),
            StateFilter(MenuState.library),
        )
        async def paginate(callback, state: FSMContext):
            page = int(callback.data.split(":", 1)[1])
            names, prev_page, next_page = self.file_library.get_page(page)
            base_index = page * self.file_library.page_size

            await callback.message.edit_text(
                self._format_entries(names, base_index),
                reply_markup=self._build_pagination(prev_page, next_page),
            )
            await callback.answer()

        @self.router.message(StateFilter(MenuState.library), F.text.regexp(r"^\d+$"))
        async def send_file(message, state: FSMContext):
            data = await state.get_data()
            files = data.get("library", [])
            index = int(message.text) - 1

            if 0 <= index < len(files):
                await self.adapter.send_file(message.from_user.id, files[index])
                return

            await self.adapter.send_message(
                message.from_user.id,
                "Unknown document number.",
            )

    async def _send_page(self, chat_id: int, names, prev_page, next_page) -> None:
        await self.adapter.send_message(
            user_id=chat_id,
            text=self._format_entries(names, 0),
            reply_markup=self._build_pagination(prev_page, next_page),
        )

    @staticmethod
    def _format_entries(names, base_index: int) -> str:
        return "\n".join(f"{base_index + i + 1}. {name}" for i, name in enumerate(names))

    def _build_pagination(self, prev_page, next_page) -> InlineKeyboardMarkup | None:
        buttons = []
        if prev_page is not None:
            buttons.append(
                InlineKeyboardButton(text=FILE_LIBRARY_PREV, callback_data=f"lib_page:{prev_page}")
            )
        if next_page is not None:
            buttons.append(
                InlineKeyboardButton(text=FILE_LIBRARY_NEXT, callback_data=f"lib_page:{next_page}")
            )
        if not buttons:
            return None
        return InlineKeyboardMarkup(inline_keyboard=[buttons])
