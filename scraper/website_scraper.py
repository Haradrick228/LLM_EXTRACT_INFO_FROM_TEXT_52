import os
import re
import random
from typing import List

import requests
import asyncio
import traceback

from dotenv import load_dotenv

from common.config import Settings
from common.util import extract_text_from_file
from core.rag.rag_service_wrapper import RagServiceWrapper

load_dotenv()

from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeout,
    Error as PlaywrightError
)
import pdfkit

from core.db_service          import DBService


db          = DBService()
settings    = Settings()
rag_wrapper = None

HEADERS           = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DOWNLOAD_EXTS     = ('.pdf', '.docx', '.pptx', '.xlsx')
BLACKLIST_DOMAINS = set()
WKHTMLTOPDF_PATH  = os.getenv("WKHTMLTOPDF_PATH")

processed_urls        = set()
_downloaded_materials = []

def is_internal(base_url: str, target_url: str) -> bool:
    return urlparse(target_url).netloc.endswith(urlparse(base_url).netloc)


def normalize_url(base: str, link: str) -> str:
    return urljoin(base, link.split('#')[0])


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^\w\-.]", "_", text, flags=re.UNICODE)[:150]


def extract_drive_id(url: str) -> str | None:
    p  = urlparse(url)
    qs = parse_qs(p.query)
    if "id" in qs and qs["id"]:
        return qs["id"][0]
    m = re.search(r"/d/([^/]+)", p.path)
    return m.group(1) if m else None


def _get_drive_confirm_token(html: str) -> str | None:
    m = re.search(r"confirm=([0-9A-Za-z_]+)&", html)
    return m.group(1) if m else None


def process_page(resource_id: int, page_url: str, file_path: str, page_title: str, headings: List[str]):
    db = DBService()
    full_text = extract_text_from_file(file_path)

    scanned_page_id = db.save_scanned_page_return_id(
        resource_id=resource_id,
        page_url=page_url,
        page_title=page_title,
        pdf_file_name=os.path.basename(file_path),
        headings=headings,
        page_text=full_text
    )
    return scanned_page_id

def download_file(resource_id: int, url: str, output_folder: str, origin_url: str):
    parsed = urlparse(url)
    dom    = parsed.netloc.lower()
    if "drive.google.com" in dom:
        fid = extract_drive_id(url)
        if not fid:
            return
        download_url = f"https://drive.google.com/uc?export=download&id={fid}"
    else:
        download_url = url

    existing = db.get_file_metadata_by_download_url(download_url)
    if existing:
        if existing['resource_id'] == resource_id:
            return
        db.save_metadata(
            resource_id=resource_id,
            origin_url=origin_url,
            download_url=download_url,
            file_name=existing['file_name'],
            skipped=True,
            file_type=existing['file_type'],
            file_size=existing['file_size']
        )
        print(f"[DL] ⏭ cross-resource skip: {download_url}")
        return

    raw_name = f"{fid}.pdf" if "drive.google.com" in dom else os.path.basename(parsed.path) or "file"
    file_name = sanitize_filename(raw_name)
    os.makedirs(output_folder, exist_ok=True)
    dst_path = os.path.join(output_folder, file_name)

    try:
        if "drive.google.com" in dom:
            session = requests.Session()
            r = session.get(download_url, headers={**HEADERS, "Referer": origin_url}, stream=True, timeout=60)
            token = _get_drive_confirm_token(r.text)
            if token:
                download_url += f"&confirm={token}"
                r = session.get(download_url, headers={**HEADERS, "Referer": origin_url}, stream=True, timeout=60)
            if "text/html" in r.headers.get("Content-Type", ""):
                download_url = f"https://drive.usercontent.google.com/u/0/uc?id={fid}&export=download"
                r = session.get(download_url, headers={**HEADERS, "Referer": origin_url}, stream=True, timeout=60)
            resp = r
        else:
            resp = requests.get(download_url, headers={**HEADERS, "Referer": origin_url}, stream=True, timeout=60)
        resp.raise_for_status()

        with open(dst_path, "wb") as f:
            for chunk in resp.iter_content(32768):
                f.write(chunk)

        file_type = os.path.splitext(file_name)[1].lstrip('.').lower()
        file_size = os.path.getsize(dst_path)

        db.save_metadata(
            resource_id=resource_id,
            origin_url=origin_url,
            download_url=download_url,
            file_name=file_name,
            skipped=False,
            file_type=file_type,
            file_size=file_size
        )
        _downloaded_materials.append(dst_path)
        print(f"[DL] downloaded: {download_url} → {dst_path}")

    except Exception as e:
        if not db.get_file_metadata_by_download_url(download_url):
            db.save_metadata(
                resource_id=resource_id,
                origin_url=origin_url,
                download_url=download_url,
                file_name=file_name,
                skipped=True
            )
        print(f"[DL] ERR {download_url}: {e}")

async def collect_all_pages(
        resource_id: int,
        base_url: str,
        output_folder: str,
        max_pages: int = 100
) -> list[tuple[str, str, str]]:
    visited = set()
    queue = [base_url]
    pages = []
    bad_subpaths = set()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=HEADERS["User-Agent"])

        while queue and len(visited) < max_pages:
            url = queue.pop(0)

            if url in visited:
                continue

            if any(bad_subpath in url for bad_subpath in bad_subpaths):
                print(f"[SKIP-SUBPATH] Пропущен проблемный под-URL: {url}")
                continue

            if url.lower().endswith(('.mp4', '.avi', '.mov')):
                print(f"[SKIP-MEDIA] Пропущен медиафайл: {url}")
                continue

            print(f"[NAV] {url}")
            visited.add(url)

            page = await context.new_page()
            try:
                await page.goto(url, timeout=80000)
                await page.wait_for_selector("body", timeout=10000)
                await asyncio.sleep(1)

                try:
                    html = await page.evaluate("document.documentElement.outerHTML")
                except PlaywrightError:
                    html = await page.content()

                try:
                    title = await page.title()
                except PlaywrightError:
                    title = None

                pages.append((url, html, title))

                try:
                    anchors = await page.query_selector_all("a")
                except PlaywrightError as e:
                    print(f"[NAV-ERR] can't query <a> on {url}: {e}")
                    anchors = []

                mat_folder = os.path.join(output_folder, "materials")
                for a in anchors:
                    href = await a.get_attribute("href") or ""
                    new = normalize_url(url, href)
                    ndom = urlparse(new).netloc.lower()

                    if (
                            "drive.google.com" in ndom
                            or new.lower().endswith(DOWNLOAD_EXTS)
                            or ndom.endswith(("yadi.sk", "disk.yandex.ru"))
                    ):
                        download_file(resource_id, new, mat_folder, origin_url=url)
                    elif is_internal(base_url, new) and new not in visited:
                        queue.append(new)

            except (PlaywrightTimeout, PlaywrightError) as e:
                print(f"[NAV-ERR] {url}: {e}")
                path = urlparse(url).path
                if path:
                    bad_subpaths.add(path)
                    print(f"[BAD-SUBPATH] Добавлен проблемный подпуть: {path}")

            except Exception as e:
                print(f"[NAV-ERR] {url}: {e}")
                traceback.print_exc()
                path = urlparse(url).path
                if path:
                    bad_subpaths.add(path)
                    print(f"[BAD-SUBPATH] Добавлен проблемный подпуть: {path}")

            finally:
                await page.close()
                await asyncio.sleep(random.uniform(1, 2))

        await browser.close()

    return pages


async def save_site_as_pdf(base_url: str, output_root: str = "scraped_site") -> list[str]:
    resource_id  = db.get_or_create_resource(base_url)
    domain       = urlparse(base_url).netloc.replace(".", "_")
    root         = os.path.join(output_root, domain)
    pages_folder = os.path.join(root, "pages")
    os.makedirs(pages_folder, exist_ok=True)

    pages = await collect_all_pages(resource_id, base_url, root)

    saved_pdfs = []
    pdf_cfg    = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH) if WKHTMLTOPDF_PATH else None

    for url, html, title in pages:
        soup = BeautifulSoup(html, "html.parser")

        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3'])]
        page_date = None

        fname    = sanitize_filename(urlparse(url).path.strip("/") or "index")
        pdf_name = f"{domain}__{fname}.pdf"
        pdf_path = os.path.join(pages_folder, pdf_name)

        text = soup.get_text(separator="\n", strip=True)
        html_doc = (
            "<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>{title or url}</title></head>"
            "<body><pre style='white-space: pre-wrap; font-family: Arial;'>"
            f"===== {url} =====\n\n{text}</pre></body></html>"
        )
        tmp = os.path.join(pages_folder, "temp.html")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(html_doc)

        if pdf_cfg:
            pdfkit.from_file(tmp, pdf_path, configuration=pdf_cfg)
        else:
            pdfkit.from_file(tmp, pdf_path)
        os.remove(tmp)

        full_text = extract_text_from_file(pdf_path)
        page_id = db.save_scanned_page_return_id(
            resource_id=resource_id,
            page_url=url,
            page_title=title,
            pdf_file_name=pdf_name,
            headings=headings,
            page_text=full_text
        )

        for a in soup.find_all("a", href=True):
            link    = normalize_url(url, a["href"])
            anchor  = a.get_text(strip=True)[:150]
            snippet = a.parent.get_text(" ", strip=True)[:200]
            db.save_page_link(
                scanned_page_id=page_id,
                download_url=link,
                anchor_text=anchor,
                context_snippet=snippet
            )

        saved_pdfs.append(pdf_path)

    all_files = list(dict.fromkeys(_downloaded_materials + saved_pdfs))
    for path in all_files:
        if not rag_wrapper.rag.is_file_indexed(path):
            rag_wrapper.rag.add_single_file(path)

    _downloaded_materials.clear()
    return saved_pdfs
