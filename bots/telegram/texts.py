"""Shared UI text constants for the Telegram bot."""

# ── Main menu buttons ──
ASK_QUESTION_BUTTON = "📝 Задать вопрос"
FILE_LIBRARY_BUTTON = "📚 Библиотека файлов"
UPLOAD_FILES_BUTTON = "📁 Загрузить файлы"
BACK_BUTTON = "🔙 Назад"

# ── Feedback / NPS ──
CANNOT_RATE_BUTTON = "❓ Не могу оценить"
SKIP_BUTTON = "🔙 Пропустить"

# ── Upload flow ──
UPLOAD_PROMPT_MESSAGE = "Отправьте PDF, DOCX или PPTX файл для загрузки."
CANCEL_BUTTON = BACK_BUTTON

# ── Model selection buttons ──
GIGACHAT_BUTTON = "🤖 GigaChat"
DEEPSEEK_BUTTON = "🧠 DeepSeek"
GIGACHAT_PRO_BUTTON = "🚀 GigaChat Pro"

# ── File library navigation ──
FILE_LIBRARY_PREV = "⟪ Назад"
FILE_LIBRARY_NEXT = "Вперёд ⟫"

# ── Feedback / NPS ──

# ── Start ──
WELCOME_MESSAGE = "Здравствуйте! Выберите, что вы хотите сделать."

# ── Back ──
BACK_RETURN_MESSAGE = "Возвращаемся в главное меню."

# ── Question flow ──
FEEDBACK_BUTTON_LABEL = "Оценить ответ"
MODEL_PROMPT = "Выберите модель для ответа:"
QUESTION_PROMPT = "Отправьте ваш вопрос текстом."
FILES_AVAILABLE_PROMPT = "Документы, найденные в базе знаний:"
FEEDBACK_PROMPT = "Помог ли вам ответ?"
MODEL_UNKNOWN_PROMPT = "Вариант не распознан. Пожалуйста, выберите одну из кнопок."
EMPTY_QUESTION_PROMPT = "Пожалуйста, отправьте текстовый вопрос."
NO_ANSWER_FALLBACK = "Ответ не был сгенерирован."

# ── Feedback flow ──
FEEDBACK_RATING_PROMPT = "Пожалуйста, оцените достоверность ответа от 1 до 10:"
FEEDBACK_COMMENT_PROMPT = "💬 Теперь напишите комментарий (или нажмите «Пропустить»):"
FEEDBACK_THANKS_WITH_COMMENT = "🙏 Спасибо за ваш комментарий и отзыв!"
FEEDBACK_THANKS_NO_COMMENT = "🙏 Спасибо за ваш отзыв!"
FEEDBACK_NPS_PROMPT = (
    "📊 Спасибо, что пользуетесь ботом! "
    "Оцените его по шкале от 0 до 10, насколько вы порекомендуете бота друзьям или коллегам:"
)
FEEDBACK_NEXT_QUESTION_PROMPT = "Если у вас есть ещё вопрос — просто напишите его."

# ── NPS flow ──
NPS_COMMENT_PROMPT = "💬 Вы можете оставить комментарий к оценке (или нажмите «🔙 Пропустить»):"
NPS_THANKS = "🙏 Спасибо за оценку работы бота!"

# ── File library ──
FILE_LIBRARY_EMPTY = "Документов пока нет."
FILE_LIBRARY_PROMPT = "Введите номер документа, чтобы получить файл."
FILE_LIBRARY_UNKNOWN_NUMBER = "Номер документа не найден."

# ── Upload ──
UPLOAD_DISABLED_TEXT = "Загрузка отключена. Укажите FOLDER_PATH в настройках."
UPLOAD_CANCELLED_MESSAGE = "Загрузка отменена. Возвращаемся в главное меню."
