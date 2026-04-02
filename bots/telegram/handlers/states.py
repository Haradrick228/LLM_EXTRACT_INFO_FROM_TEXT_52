from aiogram.fsm.state import StatesGroup, State

class MenuState(StatesGroup):
    choose_model         = State()  # выбор модели для RAG
    question             = State()  # ввод самого вопроса
    library              = State()  # просмотр файлов в библиотеке
    file                 = State()  # выбор файла по номеру
    add_link             = State()  # добавление новой ссылки
    link_library         = State()  # просмотр библиотеки ссылок
    upload               = State()  # загрузка файлов

    #Состояния для обратной связи
    waiting_for_rating   = State()  # ждем оценки 1–10
    waiting_for_comment  = State()  # ждем свободного комментария
    waiting_for_nps_rating  = State()  # ← добавили
    waiting_for_nps_comment = State()  # ← добавили