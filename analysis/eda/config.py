from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = PACKAGE_ROOT / "templates"

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".pptx")
DEFAULT_REPORT_DIR = Path("analysis") / "reports"
DEFAULT_OUTPUT_DIR = Path("processed_eda_data")
DEFAULT_TOP_TERMS = 50
TFIDF_MAX_FEATURES = 2000
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
SHORT_DOC_TOKENS = 200
LONG_DOC_TOKENS = 4000

DASHBOARD_TEMPLATE_PATH = TEMPLATE_DIR / "dashboard.html"
DASHBOARD_CSS_PATH = TEMPLATE_DIR / "dashboard.css"
