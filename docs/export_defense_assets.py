"""Export full PNG screenshots from HTML reports (no clipping/cropping)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "defense_assets"


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        eda = (ROOT / "analysis/reports/eda_dashboard.html").resolve()
        page.goto(eda.as_uri(), wait_until="networkidle")
        # Полная вкладка (архив) + отдельные секции для читаемого слайда
        page.locator("#tab-eda").screenshot(path=str(ASSETS / "eda_overview_full.png"))
        page.locator("#tab-eda section").nth(0).screenshot(path=str(ASSETS / "eda_key_figures.png"))
        page.locator("#tab-eda section").nth(1).screenshot(path=str(ASSETS / "eda_quality.png"))

        page.locator("button[data-tab='rerank-compare']").click()
        page.wait_for_timeout(400)
        page.locator("#tab-rerank-compare").screenshot(path=str(ASSETS / "fixture_rerank_compare.png"))

        canon = (ROOT / "analysis/reports/final_canonical_experiment_2026_05.html").resolve()
        page.goto(canon.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(600)
        page.locator("#panel-outcomes").screenshot(path=str(ASSETS / "canonical_outcomes.png"))

        page.locator("label[for='tab-results']").click()
        page.wait_for_timeout(400)
        page.locator("#panel-results").screenshot(path=str(ASSETS / "canonical_results.png"))

        gold = (ROOT / "analysis/reports/rag_gold_101_full.html").resolve()
        page.goto(gold.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(400)
        page.locator(".summary").screenshot(path=str(ASSETS / "latency_e2e_summary.png"))

        bench = (ROOT / "analysis/reports/embedding_benchmark_nagruzka_2026_05.html").resolve()
        page.goto(bench.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(400)
        page.locator("div[style*='display:flex']").first.screenshot(
            path=str(ASSETS / "latency_embedder_kpi.png")
        )
        # MPS-строки сводной таблицы (скорость embedder на GPU)
        page.locator("table").first.screenshot(path=str(ASSETS / "latency_embedder_table.png"))

        browser.close()
        for name in sorted(ASSETS.glob("*.png")):
            if name.name.startswith("preview"):
                continue
            print(f"OK {name}")


if __name__ == "__main__":
    main()
