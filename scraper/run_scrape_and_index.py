import os
import hashlib
from urllib.parse import urlparse

from common.config import Settings
from core.rag.rag_service import RagService
from scraper.website_scraper import save_site_as_pdf

settings = Settings()

def normalize_site_url(url: str) -> str:
    """Нормализует URL сайта до единого вида без завершающего слэша."""
    parsed = urlparse(url.strip().lower())
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    return normalized.rstrip("/")

def get_file_hash(path):
    """Возвращает SHA-256 хеш файла."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def find_all_documents(base_dirs, extensions):
    """Собирает пути ко всем документам с указанными расширениями."""
    result = []
    for base in base_dirs:
        for root, _, files in os.walk(base):
            for file in files:
                if file.lower().endswith(extensions):
                    result.append(os.path.join(root, file))
    return result

async def main():
    """Запускает скрапинг сайтов и индексацию документов в RAG."""
    rag = None
    try:
        rag = RagService(
            client_id=settings.client_id,
            client_secret=settings.client_secret,
            folder_path=settings.folder_path,
            persist_directory=settings.persist_directory,
            scope=settings.scope,
            model_name=settings.model_name
        )
    except ValueError as e:
        print(f"Индекс пока не инициализирован: {e}")

    known_sites_path = os.path.join("scraped_site", "known_sites.txt")
    if os.path.exists(known_sites_path):
        with open(known_sites_path, "r", encoding="utf-8") as f:
            raw_sites = [line.strip() for line in f if line.strip()]

        normalized_map = {}
        for original in raw_sites:
            norm = normalize_site_url(original)
            if norm not in normalized_map:
                normalized_map[norm] = original

        sites = list(normalized_map.values())

        with open(known_sites_path, "w", encoding="utf-8") as f:
            for s in sites:
                f.write(s + "\n")

        print(f"Загружено уникальных ссылок: {len(sites)} (из {len(raw_sites)} строк)")
    else:
        sites = []

    for url in sites:
        domain = urlparse(url).netloc.replace(".", "_")
        output_dir = os.path.join("scraped_site", domain)

        if os.path.exists(output_dir):
            print(f"Сайт уже обработан ранее, пропускаем: {url}")
            continue

        print(f"Обработка сайта: {url}")
        try:
            pdf_files = await save_site_as_pdf(url, output_root="scraped_site")
        except Exception as e:
            print(f"❌ Ошибка при обработке {url}: {e}")
            continue

        if rag is None:
            try:
                rag = RagService(
                    client_id=settings.client_id,
                    client_secret=settings.client_secret,
                    folder_path=settings.folder_path,
                    persist_directory=settings.persist_directory,
                    scope=settings.scope,
                    model_name=settings.model_name
                )
                print("Индекс успешно инициализирован после загрузки документов.")
            except Exception as e:
                print(f"Не удалось повторно инициализировать RagService: {e}")
                return

        for pdf_path in pdf_files:
            if not os.path.exists(pdf_path):
                continue
            file_hash = get_file_hash(pdf_path)
            if rag and (pdf_path in rag.vectordb._collection.get()["metadatas"] or file_hash in [m.get("file_hash") for m in rag.vectordb._collection.get()["metadatas"] if m]):
                print(f"Уже в индексе, пропускаем: {pdf_path}")
                continue
            print(f"📥 Индексируем: {pdf_path}")
            try:
                rag.add_single_file(pdf_path)
            except Exception as e:
                print(f"Ошибка при добавлении {pdf_path}: {e}")

    folders = ["scraped_site", "gigachat_materials"] + [
        f for f in os.listdir() if f.startswith("gigachat_") and os.path.isdir(f)
    ]

    all_docs = find_all_documents(folders, extensions=(".pdf", ".docx", ".pptx"))
    print(f"Найдено документов для индексации: {len(all_docs)}")

    new_count = 0
    skipped_count = 0

    if rag is None:
        try:
            rag = RagService(
                client_id=settings.client_id,
                client_secret=settings.client_secret,
                folder_path=settings.folder_path,
                persist_directory=settings.persist_directory,
                scope=settings.scope,
                model_name=settings.model_name
            )
            print("Индекс успешно инициализирован перед индексированием файлов.")
        except Exception as e:
            print(f"Не удалось инициализировать RagService: {e}")
            return

    for path in all_docs:
        if not os.path.exists(path):
            continue
        file_hash = get_file_hash(path)
        if path in rag.vectordb._collection.get()["metadatas"] or file_hash in [m.get("file_hash") for m in rag.vectordb._collection.get()["metadatas"] if m]:
            print(f"Уже в индексе, пропускаем: {path}")
            skipped_count += 1
            continue
        print(f"Индексируем: {path}")
        try:
            rag.add_single_file(path)
            new_count += 1
        except Exception as e:
            print(f"Ошибка при добавлении {path}: {e}")

    print("\nИтоговая статистика:")
    print(f"  - Обработано сайтов: {len(sites)}")
    print(f"  - Всего проиндексировано новых файлов: {new_count}")
    print(f"  - Пропущено (уже в индексе): {skipped_count}")
    print("Полная индексация завершена.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
