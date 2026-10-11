"""Generate defense_presentation.pptx — commission-ready edition (14 slides).

Rebuild:
    python docs/export_defense_assets.py
    python docs/generate_defense_presentation.py

Optional: place telegram_demo.png in docs/defense_assets/ for slide 12 screenshot.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

DOCS = Path(__file__).resolve().parent
NUMBERS = DOCS / "defense_verified_numbers.json"
OUT = DOCS / "defense_presentation.pptx"
OUT_HUMAN = DOCS / "defense_presentation_human.pptx"
OUT_FALLBACK = DOCS / "defense_presentation_styled.pptx"
ASSETS = DOCS / "defense_assets"
PREVIEW = ASSETS / "preview"

SLIDE_W = Inches(10)
SLIDE_H = Inches(7.5)
ML = 0.65
CONTENT_W = 8.7
FOOTER_Y = 7.08

C_TEXT = RGBColor(35, 35, 35)
C_MUTED = RGBColor(110, 110, 110)
C_LINE = RGBColor(210, 210, 210)
C_HDR = RGBColor(250, 250, 250)
C_LEAD = RGBColor(55, 55, 55)
C_BOX = RGBColor(245, 247, 252)
C_BOX_BORDER = RGBColor(180, 190, 210)
C_ARROW = RGBColor(90, 100, 130)

EMBEDDER_LABELS = [
    "mpnet (multilingual)",
    "BGE-M3",
    "FRIDA",
    "E5-large-instruct",
    "Jina-embeddings-v3",
    "default_clean (legacy)",
    "USER-bge-m3",
]

MATRIX_TOP = [
    ("USER-bge-m3", "Jina reranker", "6,4%"),
    ("FRIDA", "Jina reranker", "6,4%"),
    ("FRIDA", "BGE reranker", "6,4%"),
    ("E5-large-instruct", "BGE reranker", "5,9%"),
    ("Jina-embeddings-v3", "BGE reranker", "5,9%"),
    ("USER-bge-m3", "BGE reranker", "5,4%"),
    ("BGE-M3", "BGE reranker", "5,4%"),
    ("mpnet", "MiniLM (baseline)", "2,0%"),
]


def load_numbers() -> dict:
    with NUMBERS.open(encoding="utf-8") as f:
        return json.load(f)


class Deck:
    def __init__(self, prs: Presentation) -> None:
        self.prs = prs
        self.n = 0

    def slide(self):
        self.n += 1
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = RGBColor(255, 255, 255)
        return s

    def title_block(self, slide, title: str, lead: str = "", *, y: float = 0.42) -> float:
        tb = slide.shapes.add_textbox(Inches(ML), Inches(y), Inches(CONTENT_W), Inches(0.5))
        p = tb.text_frame.paragraphs[0]
        p.text = title
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = C_TEXT
        p.font.name = "Calibri"
        ny = y + 0.52
        if lead:
            lb = slide.shapes.add_textbox(Inches(ML), Inches(ny), Inches(CONTENT_W), Inches(0.45))
            lp = lb.text_frame.paragraphs[0]
            lp.text = lead
            lp.font.size = Pt(12)
            lp.font.color.rgb = C_LEAD
            lp.font.name = "Calibri"
            ny += 0.42
        line = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(ML), Inches(ny), Inches(CONTENT_W), Inches(0.01)
        )
        line.fill.solid()
        line.fill.fore_color.rgb = C_LINE
        line.line.fill.background()
        return ny + 0.22

    def footer(self, slide) -> None:
        tb = slide.shapes.add_textbox(Inches(ML), Inches(FOOTER_Y), Inches(7), Inches(0.28))
        tb.text_frame.paragraphs[0].text = "Годовой проект ВШЭ 2025–2026 · Серов Арсений"
        tb.text_frame.paragraphs[0].font.size = Pt(9)
        tb.text_frame.paragraphs[0].font.color.rgb = C_MUTED
        num = slide.shapes.add_textbox(Inches(9.15), Inches(FOOTER_Y), Inches(0.45), Inches(0.28))
        num.text_frame.paragraphs[0].text = str(self.n)
        num.text_frame.paragraphs[0].font.size = Pt(9)
        num.text_frame.paragraphs[0].font.color.rgb = C_MUTED
        num.text_frame.paragraphs[0].alignment = PP_ALIGN.RIGHT

    def bullets(self, slide, items: list[str], *, top: float, size: int = 15, spacing: int = 11) -> float:
        tb = slide.shapes.add_textbox(Inches(ML), Inches(top), Inches(CONTENT_W), Inches(FOOTER_Y - top - 0.1))
        tf = tb.text_frame
        tf.word_wrap = True
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text = item
            p.font.size = Pt(size)
            p.font.color.rgb = C_TEXT
            p.font.name = "Calibri"
            p.space_after = Pt(spacing)
            p.line_spacing = 1.2
        lines = sum(max(1, len(item) // 95 + 1) for item in items)
        return top + lines * (size / 72 * 1.25) + (len(items) - 1) * spacing / 72

    def note(self, slide, text: str, *, top: float, size: int = 11) -> None:
        tb = slide.shapes.add_textbox(Inches(ML), Inches(top), Inches(CONTENT_W), Inches(0.55))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(size)
        p.font.color.rgb = C_MUTED
        p.font.name = "Calibri"
        p.line_spacing = 1.15

    def table(
        self,
        slide,
        headers: list[str],
        rows: list[list[str]],
        *,
        top: float,
        col_widths: list[float],
        font_size: int = 11,
    ) -> None:
        rh = 0.36
        shape = slide.shapes.add_table(
            len(rows) + 1, len(headers), Inches(ML), Inches(top), Inches(CONTENT_W), Inches(rh * (len(rows) + 1))
        )
        tbl = shape.table
        for j, w in enumerate(col_widths):
            tbl.columns[j].width = Inches(w)
        for j, h in enumerate(headers):
            c = tbl.cell(0, j)
            c.text = h
            c.fill.solid()
            c.fill.fore_color.rgb = C_HDR
            for p in c.text_frame.paragraphs:
                p.font.bold = True
                p.font.size = Pt(font_size)
                p.font.color.rgb = C_TEXT
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
        for i, row in enumerate(rows, 1):
            for j, val in enumerate(row):
                c = tbl.cell(i, j)
                c.text = val
                for p in c.text_frame.paragraphs:
                    p.font.size = Pt(font_size)
                    p.font.color.rgb = C_TEXT
                c.vertical_anchor = MSO_ANCHOR.MIDDLE

    def picture_fit(
        self,
        slide,
        filename: str,
        *,
        top: float,
        left: float = ML,
        max_w: float | None = None,
        max_h: float | None = None,
    ) -> float:
        path = ASSETS / filename
        if not path.is_file():
            return top + 0.3
        mw = max_w if max_w is not None else CONTENT_W
        mh = max_h if max_h is not None else FOOTER_Y - top - 0.15
        with Image.open(path) as im:
            px_w, px_h = im.size
        aspect = px_w / px_h
        w_by_width = mw
        h_by_width = w_by_width / aspect
        if h_by_width <= mh:
            w, h = w_by_width, h_by_width
        else:
            h = mh
            w = h * aspect
        slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(w))
        return top + h

    def cp_section(self, slide, heading: str, facts: list[str], *, top: float) -> float:
        est_h = 0.26 + len(facts) * 0.23
        tb = slide.shapes.add_textbox(Inches(ML), Inches(top), Inches(CONTENT_W), Inches(est_h))
        tf = tb.text_frame
        tf.word_wrap = True
        hp = tf.paragraphs[0]
        hp.text = heading
        hp.font.size = Pt(13)
        hp.font.bold = True
        hp.font.color.rgb = C_TEXT
        hp.font.name = "Calibri"
        hp.space_after = Pt(3)
        for fact in facts:
            fp = tf.add_paragraph()
            fp.text = fact
            fp.font.size = Pt(11)
            fp.font.color.rgb = C_TEXT
            fp.font.name = "Calibri"
            fp.space_after = Pt(4)
            fp.line_spacing = 1.15
        return top + est_h + 0.05

    def _box(self, slide, left: float, top: float, w: float, h: float, text: str, *, size: int = 9) -> None:
        shape = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(w), Inches(h)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = C_BOX
        shape.line.color.rgb = C_BOX_BORDER
        shape.line.width = Pt(0.75)
        tf = shape.text_frame
        tf.word_wrap = True
        tf.margin_left = Pt(4)
        tf.margin_right = Pt(4)
        tf.margin_top = Pt(3)
        tf.margin_bottom = Pt(3)
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(size)
        p.font.color.rgb = C_TEXT
        p.font.name = "Calibri"
        p.alignment = PP_ALIGN.CENTER
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    def _arrow_right(self, slide, x: float, y: float, length: float = 0.28) -> None:
        conn = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y), Inches(x + length), Inches(y)
        )
        conn.line.color.rgb = C_ARROW
        conn.line.width = Pt(1.25)

    def architecture_diagram(self, slide, *, top: float) -> float:
        box_h = 0.68
        gap = 0.32
        x0 = ML
        row1 = [
            ("Пользователь\nпедагог / родитель /\nсотрудник фонда", 1.35),
            ("Telegram-бот\nREST API :8001", 1.35),
            ("RagService\nml_service :8000", 1.35),
            ("Chroma + BM25\n7 коллекций · 61 543 фраг.", 1.65),
        ]
        x = x0
        for i, (text, w) in enumerate(row1):
            self._box(slide, x, top, w, box_h, text, size=8)
            if i < len(row1) - 1:
                self._arrow_right(slide, x + w + 0.02, top + box_h / 2, gap - 0.06)
            x += w + gap
        row2_y = top + box_h + 0.42
        row2 = [
            ("Cross-encoder reranker\nBGE-v2-m3", 1.55),
            ("LLM\nDeepSeek / Ollama / GigaChat", 1.55),
            ("PostgreSQL\nлоги · NPS · feedback", 1.55),
        ]
        x2 = x0 + 1.7
        chroma_center = x0 + 1.35 + 0.32 + 1.35 + 0.32 + 1.35 + 0.825
        conn = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT,
            Inches(chroma_center),
            Inches(top + box_h),
            Inches(chroma_center),
            Inches(row2_y),
        )
        conn.line.color.rgb = C_ARROW
        conn.line.width = Pt(1.25)
        for i, (text, w) in enumerate(row2):
            self._box(slide, x2, row2_y, w, box_h, text, size=8)
            if i < len(row2) - 1:
                self._arrow_right(slide, x2 + w + 0.02, row2_y + box_h / 2, gap - 0.06)
            x2 += w + gap
        eval_y = row2_y + box_h + 0.28
        self._box(
            slide,
            ML,
            eval_y,
            CONTENT_W,
            0.58,
            "Eval (параллельно): скрипты прогона · frozen protocol · 7×3 матрица · "
            "fetch_k=40 · JSON-трейсы · HTML-отчёты",
            size=9,
        )
        return eval_y + 0.62


def build() -> Path:
    n = load_numbers()
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    d = Deck(prs)

    gold = n["corpus"]["gold_questions"]
    chunks = f"{n['corpus']['chunks']:,}".replace(",", " ")
    docs = n["eda"]["documents"]
    dup = n["eda"]["duplicate_rate_pct"]
    ru_pct = n["eda"]["dominant_language_ru_pct"]
    max_r10 = n["full_corpus_substring_recall_at_10"]["max_pct"]
    prod_r10 = n["production_recommendation"]["retrieval_r_at_10_pct"]
    squad = n["validity"]["squad_sanity_r_at_10_pct"]
    neg_h = n["validity"]["neg_control_honest_pct"]
    neg_s = n["validity"]["neg_control_shuffled_pct"]
    class_a = n["validity"]["metrics_ladder_class_a_pct_approx"]
    ans_sim = n["answer_eval"]["best_avg_ref_sim"]
    lat = n.get("latency", {}).get("e2e_prod_stack", {})
    e2e_avg = lat.get("avg_time_s", 1.9)
    e2e_med = lat.get("median_time_s", 1.7)
    e2e_p95 = lat.get("p95_time_s", 3.0)
    e2e_max = lat.get("max_time_s", 8.0)
    emb_bench = n.get("latency", {}).get("embedder_benchmark_mps_medium_s", {})
    emb_mem = n.get("latency", {}).get("embedder_memory_mb", {})

    # 1. Титул
    s = d.slide()
    tb = s.shapes.add_textbox(Inches(ML), Inches(1.35), Inches(CONTENT_W), Inches(1.35))
    tb.text_frame.paragraphs[0].text = "Применение LLM для извлечения\nинформации из текстов"
    tb.text_frame.paragraphs[0].font.size = Pt(28)
    tb.text_frame.paragraphs[0].font.bold = True
    tb.text_frame.paragraphs[0].font.color.rgb = C_TEXT
    sub = s.shapes.add_textbox(Inches(ML), Inches(2.75), Inches(CONTENT_W), Inches(2.8))
    sf = sub.text_frame
    sf.word_wrap = True
    for i, line in enumerate(
        [
            "RAG-ассистент для благотворительного фонда «Вклад в будущее»",
            "и исследование лучших комбинаций embedder × reranker на полном корпусе",
            "",
            "Годовой проект ВШЭ, направление «Искусственный интеллект»",
            "Серов Арсений Алексеевич · Научный руководитель: Малюшитский Кирилл",
            "2025–2026",
        ]
    ):
        sp = sf.paragraphs[0] if i == 0 else sf.add_paragraph()
        sp.text = line
        sp.font.size = Pt(13 if i < 3 else 14)
        sp.font.color.rgb = C_LEAD if i < 3 else C_MUTED
        sp.space_after = Pt(4)
    d.footer(s)

    # 2. Проблема и цель
    s = d.slide()
    y = d.title_block(
        s,
        "Проблема и цель проекта",
        "База знаний благотворительной организации — ответы без ручного поиска по сотням PDF.",
    )
    d.bullets(
        s,
        [
            "Заказчик по смыслу данных — БФ «Вклад в будущее»: методички, программы, "
            "курсы и справочные материалы с books.vbudushee.ru.",
            "Проблема: сотруднику, педагогу или родителю сложно быстро найти ответ "
            "в разрозненных документах по инклюзивной среде, развитию детей, "
            "обучению педагогов и цифровым/ИИ-курсам.",
            "Решение — RAG: поиск релевантных фрагментов в индексе, затем ответ языковой "
            "модели строго по найденному контексту.",
            "Цель года — два результата: работающий сервис (Telegram-бот + API) "
            f"и исследование: какая связка моделей лучше на {chunks} фрагментах "
            f"и {gold} контрольных вопросах.",
            "Сознательно не делали NER и GraphRAG — фокус на качестве поиска, "
            "честной оценке и воспроизводимом эксперименте 7×3.",
        ],
        top=y + 0.03,
        size=13,
        spacing=8,
    )
    d.footer(s)

    # 3. Пользователи и сценарий
    s = d.slide()
    y = d.title_block(
        s,
        "Кому помогает система",
        "Типичный сценарий: вопрос обычным языком — ответ по всей базе знаний организации.",
    )
    d.bullets(
        s,
        [
            "Педагоги и воспитатели — методические пособия, игры, работа с детьми "
            "5–18 лет, в том числе с ОВЗ.",
            "Родители — программы «Развивающая среда», «Школа возможностей», "
            "Хрестоматия, практики открытости.",
            "Сотрудники и партнёры фонда — справки о программах, миссии, "
            "финансовой грамотности, цифровых и ИИ-инициативах.",
            "Пример вопроса: «Что такое серия образовательных материалов "
            "„Школа возможностей“?» — RAG-бот находит фрагменты и формирует связный ответ.",
            "Вопрос вне базы — система честно сообщает, что данных недостаточно "
            "(не выдумывает факты).",
        ],
        top=y + 0.03,
        size=13,
        spacing=9,
    )
    d.footer(s)

    # 4. Данные
    s = d.slide()
    y = d.title_block(
        s,
        "Данные: что внутри корпуса",
        f"{docs} документов фонда → {chunks} поисковых фрагментов в индексе.",
    )
    text_end = d.bullets(
        s,
        [
            "Источник — образовательные материалы фонда: программа «Развивающая среда», "
            "серия «Школа возможностей», Хрестоматия по возрастам, методические пособия.",
            "Аудитория материалов: дети 5–18 лет, педагоги, родители, управленцы; "
            "темы — инклюзивная среда, личностное развитие, финграмотность, ИИ и цифровые навыки.",
            f"После очистки: {n['eda']['unique_texts']} уникальных текстов, "
            f"язык в основном русский ({ru_pct}%), дубликаты {dup}%.",
            "Нарезка 650 символов с перекрытием 250 — выбрана экспериментом "
            "(альтернативные режимы не улучшили метрики).",
        ],
        top=y + 0.02,
        size=12,
        spacing=7,
    )
    img_top = text_end + 0.12
    half = (CONTENT_W - 0.25) / 2
    d.picture_fit(s, "eda_key_figures.png", top=img_top, left=ML, max_w=half, max_h=FOOTER_Y - img_top - 0.12)
    d.picture_fit(
        s, "eda_quality.png", top=img_top, left=ML + half + 0.25, max_w=half, max_h=FOOTER_Y - img_top - 0.12
    )
    d.footer(s)

    # 5. Архитектура
    s = d.slide()
    y = d.title_block(
        s,
        "Архитектура системы",
        "Поток запроса от пользователя до ответа и логирования.",
    )
    diag_end = d.architecture_diagram(s, top=y + 0.05)
    d.note(
        s,
        "В боте: USER-bge-m3 + BGE reranker, fetch_k=20, top_k=5. "
        "В эксперименте: 7 моделей, fetch_k=40, матрица 7×3.",
        top=diag_end + 0.06,
        size=10,
    )
    d.footer(s)

    # 6. Исследование 7×3
    s = d.slide()
    y = d.title_block(
        s,
        "Исследовательская часть: матрица 7×3",
        "Систематическое сравнение лучших open-source связок на полном корпусе.",
    )
    d.bullets(
        s,
        [
            f"21 конфигурация: 7 embedding-моделей × 3 cross-encoder reranker, "
            f"{gold} вопросов, {chunks} фрагментов, 2 121 оценка.",
            "Embedder-индексы (отдельная коллекция Chroma на каждую модель):",
        ],
        top=y + 0.02,
        size=12,
        spacing=6,
    )
    emb_top = y + 0.95
    col_w = CONTENT_W / 2 - 0.1
    for i, label in enumerate(EMBEDDER_LABELS):
        col = i // 4
        row = i % 4
        bx = ML + col * (col_w + 0.2)
        by = emb_top + row * 0.38
        d._box(s, bx, by, col_w, 0.32, f"• {label}", size=9)
    extra_top = emb_top + 4 * 0.38 + 0.08
    d.bullets(
        s,
        [
            "Reranker: MiniLM (baseline), BGE-v2-m3, Jina-v3.",
            "Дополнительно исследовали: BM25, hybrid RRF, HyDE, multi-vector BGE-M3, "
            "смену режимов чанкования.",
            f"Лучший поиск по substring Recall@10: {max_r10}% "
            "(FRIDA или USER-bge-m3 + BGE или Jina reranker).",
        ],
        top=extra_top,
        size=11,
        spacing=7,
    )
    d.footer(s)

    # 7. Этапы — данные и поиск
    s = d.slide()
    y = d.title_block(s, "Этапы: данные и поиск", "CP2, CP3, CP5 — от сырья до готовых индексов.")
    y = d.cp_section(
        s,
        "CP2. EDA и подготовка",
        [
            f"Обработали {docs} файлов фонда, HTML-дашборд, TF-IDF, контроль качества загрузки.",
            f"Зафиксировали язык ({ru_pct}% RU), дубликаты, проблемные PDF; экспорт чанков для индексации.",
        ],
        top=y + 0.02,
    )
    y = d.cp_section(
        s,
        "CP3. Baseline RAG",
        [
            f"Собрали {gold} контрольных вопросов с эталонными цитатами из материалов фонда.",
            f"Первый прогон mpnet + MiniLM на полной базе: ~"
            f"{n['full_corpus_substring_recall_at_10']['baseline_mpnet_max_pct']}% "
            "дословного попадания в top-10.",
        ],
        top=y,
    )
    d.cp_section(
        s,
        "CP5. Улучшение поиска",
        [
            "7 коллекций Chroma, BM25 sidecar, hybrid / HyDE / multi-vector.",
            "Подготовили воспроизводимые прогоны (replay_experiment) для матрицы 7×3.",
        ],
        top=y,
    )
    d.footer(s)

    # 8. Этапы — сервис и эксперимент
    s = d.slide()
    y = d.title_block(s, "Этапы: сервис и финальный эксперимент", "CP4, CP6, CP7.")
    y = d.cp_section(
        s,
        "CP4. Продукт",
        [
            "Telegram-бот: меню, выбор LLM, feedback, NPS, загрузка материалов, библиотека файлов.",
            "REST API :8001, ml_service :8000, история запросов в PostgreSQL.",
        ],
        top=y + 0.02,
    )
    y = d.cp_section(
        s,
        "CP6. Финальный эксперимент",
        [
            f"Матрица 7×3, frozen protocol, canonical-отчёт и HTML-дашборд.",
            f"Максимум поиска: {max_r10}% substring Recall@10.",
        ],
        top=y,
    )
    d.cp_section(
        s,
        "CP7. Диагностика",
        [
            f"Трейсы на каждый запрос; neg-control {neg_h}%→{neg_s}%; SQuAD {squad}%.",
            "HTML-отчёт с разбором худших запросов и типами ошибок.",
        ],
        top=y,
    )
    d.footer(s)

    # 9. Методология
    s = d.slide()
    y = d.title_block(
        s,
        "Как оценивали качество",
        "Две задачи: нашли ли нужный фрагмент и насколько хорош текст ответа.",
    )
    d.bullets(
        s,
        [
            f"{gold} контрольных вопросов с эталонными цитатами из материалов фонда.",
            "Главная метрика поиска: эталонная фраза входит в top-10 найденных фрагментов "
            "(строгая проверка, близкая к дословному совпадению).",
            f"Качество ответа LLM — отдельно: близость к эталонному ответу эксперта "
            f"(лучший результат {ans_sim:.2f}).",
            f"Проверка корректности: SQuAD sanity {squad}%; "
            f"при перемешивании эталонов метрика падает с {neg_h}% до {neg_s}%.",
            f"~{class_a}% вопросов (class A): эталона нет в индексе как подстрока — "
            "это потолок данных, не баг одной модели.",
        ],
        top=y + 0.03,
        size=12,
        spacing=9,
    )
    d.footer(s)

    # 10. Результаты
    s = d.slide()
    y = d.title_block(
        s,
        "Что получилось",
        f"Итоги canonical-эксперимента на полной базе ({chunks} фрагментов).",
    )
    d.table(
        s,
        ["Вариант", "Поиск, %", "Ответ", "Зачем"],
        [
            ["Старт: mpnet + MiniLM", "до 2", "—", "Baseline CP3"],
            ["Максимум поиска", f"{max_r10}", "0,65", "FRIDA / USER-bge + Jina или BGE"],
            ["Production (бот)", f"{prod_r10}", "0,64", "USER-bge-m3 + BGE reranker"],
            ["Лучший текст ответа", "5,4", f"{ans_sim:.2f}".replace(".", ","), "BGE-M3 + BGE reranker"],
        ],
        top=y + 0.05,
        col_widths=[2.5, 1.0, 0.85, 4.35],
        font_size=10,
    )
    d.bullets(
        s,
        [
            f"Поиск вырос с 2% до {max_r10}%. Низкий процент — строгая метрика "
            f"и сложный корпус (~{class_a}% эталонов изначально отсутствуют в индексе).",
            "В боте USER-bge-m3 + BGE (5,4%), а не 6,4%: быстрый индекс, стабильная пара, "
            "близость ответа 0,642.",
            "Цифры 64% и 95% на ранних дашбордах — маленький тестовый набор, не полная база.",
        ],
        top=y + 1.95,
        size=12,
        spacing=9,
    )
    d.footer(s)

    # 11. Скорость ответа
    s = d.slide()
    y = d.title_block(
        s,
        "Скорость ответа и нагрузка",
        "Метрики из HTML-отчётов: полный прогон 101 вопроса и бенчмарк embedder.",
    )
    d.table(
        s,
        ["Уровень", "Конфигурация", "Метрика", "Значение"],
        [
            [
                "Полный ответ (бот)",
                "USER-bge-m3 + BGE + qwen2.5:7b",
                "Среднее / медиана / p95",
                f"{e2e_avg}s / {e2e_med}s / {e2e_p95}s",
            ],
            [
                "Полный ответ (бот)",
                f"{lat.get('questions', gold)} вопросов, 61 543 фрагмента",
                "Максимум",
                f"{e2e_max}s",
            ],
            [
                "Только embedder (MPS)",
                "USER-bge-m3",
                "encode medium",
                f"{emb_bench.get('USER-bge-m3', 0.046)}s · {emb_mem.get('USER-bge-m3', 1370)} MB",
            ],
            [
                "Только embedder (MPS)",
                "FRIDA (max recall)",
                "encode medium",
                f"{emb_bench.get('FRIDA', 0.204)}s · {emb_mem.get('FRIDA', 3141)} MB",
            ],
            [
                "Только embedder (MPS)",
                "BGE-M3",
                "encode medium",
                f"{emb_bench.get('bge-m3', 0.127)}s · {emb_mem.get('bge-m3', 2166)} MB",
            ],
        ],
        top=y + 0.04,
        col_widths=[1.35, 2.5, 1.45, 3.4],
        font_size=9,
    )
    img_top = y + 2.05
    half = (CONTENT_W - 0.2) / 2
    d.picture_fit(
        s,
        "latency_e2e_summary.png",
        top=img_top,
        left=ML,
        max_w=half,
        max_h=FOOTER_Y - img_top - 0.12,
    )
    d.picture_fit(
        s,
        "latency_embedder_kpi.png",
        top=img_top,
        left=ML + half + 0.2,
        max_w=half,
        max_h=FOOTER_Y - img_top - 0.12,
    )
    d.note(
        s,
        "Источники: rag_gold_101_full.html (e2e) и embedding_benchmark_nagruzka_2026_05.html. "
        "USER-bge-m3 в prod — баланс скорости индекса и 5,4% recall.",
        top=FOOTER_Y - 0.38,
        size=9,
    )
    d.footer(s)

    # 12. Топ комбинаций (читаемая таблица)
    s = d.slide()
    y = d.title_block(
        s,
        "Лучшие комбинации embedder + reranker",
        "Substring Recall@10 на полном корпусе. Полная таблица 21 ячейка — в HTML-отчёте.",
    )
    d.table(
        s,
        ["#", "Embedder", "Reranker", "Recall@10"],
        [[str(i), emb, rer, pct] for i, (emb, rer, pct) in enumerate(MATRIX_TOP, 1)],
        top=y + 0.06,
        col_widths=[0.4, 2.8, 2.5, 1.0],
        font_size=10,
    )
    d.note(
        s,
        "Подсветка 6,4% — три лучших комбо. Production: USER-bge-m3 + BGE (5,4%) — "
        "баланс скорости индекса и качества.",
        top=y + 3.55,
        size=10,
    )
    d.footer(s)

    # 13. Демо
    s = d.slide()
    y = d.title_block(s, "Демонстрация на защите", "Живой Telegram-бот и пример ответа.")
    demo_left_w = 4.0
    img_path = ASSETS / "telegram_demo.png"
    if img_path.is_file():
        d.picture_fit(s, "telegram_demo.png", top=y + 0.05, left=ML, max_w=demo_left_w, max_h=FOOTER_Y - y - 0.35)
    else:
        d._box(
            s,
            ML,
            y + 0.05,
            demo_left_w,
            2.8,
            "Скрин Telegram-бота\n(telegram_demo.png)\n\nНа защите — живой показ",
            size=11,
        )
    ex = s.shapes.add_textbox(Inches(ML + demo_left_w + 0.3), Inches(y + 0.05), Inches(4.4), Inches(3.2))
    tf = ex.text_frame
    tf.word_wrap = True
    lines = [
        ("Вопрос:", True),
        ("Что такое «Школа возможностей»?", False),
        ("", False),
        ("Ответ (сокращённо):", True),
        (
            "Серия образовательных материалов фонда «Вклад в будущее» "
            "в рамках программы «Развивающая среда». Готовит детей 5–18 лет "
            "к осознанному выбору, вовлекает педагогов и родителей.",
            False,
        ),
        ("", False),
        ("Также покажу:", True),
        ("• вопрос вне базы → «недостаточно данных»", False),
        ("• переключение профиля RAG", False),
        ("• HTML-отчёт с ошибками поиска", False),
    ]
    for i, (text, bold) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.font.size = Pt(11)
        p.font.bold = bold
        p.font.color.rgb = C_TEXT
        p.font.name = "Calibri"
        p.space_after = Pt(3)
    d.footer(s)

    # 14. Выводы
    s = d.slide()
    y = d.title_block(s, "Главные выводы", "")
    d.bullets(
        s,
        [
            "Для фонда: RAG-бот даёт быстрый доступ к знаниям организации — "
            "педагог, родитель или сотрудник получает ответ по всей базе, а не по одному PDF.",
            f"Для исследования: систематически сравнили 7×3=21 конфигурацию; "
            f"поиск вырос с 2% до {max_r10}%; выбрана prod-пара USER-bge-m3 + BGE ({prod_r10}%).",
            f"Методология честная: ~{class_a}% эталонов не в индексе; "
            f"SQuAD {squad}% и neg-control {neg_h}%→{neg_s}% подтверждают корректность оценки.",
            f"Сданы: Telegram-бот, REST API, frozen eval protocol, "
            "canonical HTML-отчёт, воспроизводимые прогоны.",
        ],
        top=y + 0.02,
        size=13,
        spacing=10,
    )
    thanks = s.shapes.add_textbox(Inches(ML), Inches(6.55), Inches(CONTENT_W), Inches(0.35))
    thanks.text_frame.paragraphs[0].text = "Спасибо за внимание. Готов ответить на вопросы и показать бота."
    thanks.text_frame.paragraphs[0].font.size = Pt(12)
    thanks.text_frame.paragraphs[0].font.color.rgb = C_MUTED
    d.footer(s)

    targets = [OUT, OUT_HUMAN]
    saved = OUT
    for target in targets:
        try:
            prs.save(target)
            saved = target
        except PermissionError:
            continue
    if saved == OUT:
        try:
            prs.save(OUT_HUMAN)
        except PermissionError:
            pass
    else:
        try:
            prs.save(OUT_FALLBACK)
            saved = OUT_FALLBACK
        except PermissionError:
            pass
    return saved


def export_slide_previews(pptx_path: Path) -> list[Path]:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    for old in PREVIEW.glob("slide_*.png"):
        old.unlink()
    import win32com.client  # type: ignore

    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = 1
    pres = app.Presentations.Open(str(pptx_path.resolve()), WithWindow=False)
    out: list[Path] = []
    try:
        for i in range(1, pres.Slides.Count + 1):
            p = PREVIEW / f"slide_{i:02d}.png"
            pres.Slides(i).Export(str(p.resolve()), "PNG", 1920, 1080)
            out.append(p)
    finally:
        pres.Close()
        app.Quit()
    return out


def ocr_qa_previews(preview_dir: Path) -> list[dict]:
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    expectations: dict[int, list[str]] = {
        1: ["RAG", "embedder", "2025"],
        2: ["RAG", "PDF", "books"],
        3: ["RAG", "5", "18"],
        4: ["973", "650", "543"],
        5: ["Telegram", "Chroma", "RagService"],
        6: ["21", "embedder", "reranker"],
        7: ["CP2", "CP3", "Chroma"],
        8: ["CP4", "Telegram", "CP6"],
        9: ["101", "SQuAD", "class"],
        10: ["USER-bge", "6,4", "production"],
        11: ["1.9", "USER-bge", "FRIDA"],
        12: ["FRIDA", "6,4", "USER-bge"],
        13: ["Telegram", "RAG", "HTML"],
        14: ["SQuAD", "7", "API"],
    }
    results: list[dict] = []
    for png in sorted(preview_dir.glob("slide_*.png")):
        idx = int(png.stem.split("_")[1])
        text = pytesseract.image_to_string(Image.open(png), lang="rus+eng")
        clean = re.sub(r"\s+", " ", text).strip()
        words = len(clean.split())
        missing = [kw for kw in expectations.get(idx, []) if kw.lower() not in clean.lower()]
        ok = len(clean) > 40 and not missing and (idx == 1 or words >= 18)
        results.append({"slide": idx, "ok": ok, "words": words, "missing": missing})
    return results


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
    try:
        previews = export_slide_previews(path)
        print(f"Exported {len(previews)} previews")
        qa = ocr_qa_previews(PREVIEW)
        for r in qa:
            st = "OK" if r["ok"] else "FAIL"
            extra = f" missing={r['missing']}" if r["missing"] else ""
            print(f"  slide {r['slide']:02d}: {st} ({r['words']} words){extra}")
        if not all(r["ok"] for r in qa):
            print("OCR QA: some slides flagged")
            sys.exit(1)
        print("OCR QA: all passed")
    except Exception as exc:
        print(f"Preview/OCR skipped: {exc}")
