import os
import traceback
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from chromadb import HttpClient

os.environ["CHROMA_API_IMPL"]         = "rest"
os.environ["CHROMA_SERVER_HOST"]      = os.getenv("CHROMA_SERVER_HOST", "localhost")
os.environ["CHROMA_SERVER_HTTP_PORT"] = os.getenv("CHROMA_SERVER_HTTP_PORT", "8000")
os.environ["CHROMA_COLLECTION_NAME"]  = os.getenv("CHROMA_COLLECTION_NAME", "default")

client = HttpClient()
vectordb = Chroma(
    client=client,
    embedding_function=HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ),
    collection_name=os.environ["CHROMA_COLLECTION_NAME"],
)

project_root = Path(__file__).resolve().parent
scraped       = project_root / "scraped_site"

folders = []
if scraped.is_dir():
    for domain in sorted(scraped.iterdir()):
        if not domain.is_dir():
            continue
        p = domain / "pages"
        m = domain / "materials"
        if p.is_dir():
            folders.append(p)
        if m.is_dir():
            folders.append(m)


print("=== Папки для сканирования ===")
for f in folders:
    print(f"  • {f}  (exists: {f.exists()})")

to_index = []
for folder in folders:
    for file in folder.rglob("*"):
        if file.is_file() and file.suffix.lower().lstrip(".") in ("pdf", "docx", "pptx"):
            to_index.append(file)

print(f"Найдено файлов для индексации: {len(to_index)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=650,
    chunk_overlap=250,
    separators=["\n\n", "\n", "。", "!", "?", "]】", ")", "}", "›"],
)

for path in to_index:
    print(f"\n>>> Обработка {path}")
    ext = path.suffix.lower()[1:]
    try:
        if ext == "pdf":
            loader = PyPDFLoader(str(path))
        elif ext == "docx":
            loader = Docx2txtLoader(str(path))
        else:
            loader = UnstructuredPowerPointLoader(str(path))
        docs = loader.load()
    except Exception as e:
        print(f"⚠️ Не удалось загрузить {path.name}: {e}")
        traceback.print_exc(limit=1)
        continue

    docs = [d for d in docs if d.page_content and d.page_content.strip()]
    if not docs:
        print(f"⚠️ Нет текста в {path.name}, пропускаем.")
        continue

    try:
        chunks = splitter.split_documents(docs)
    except Exception as e:
        print(f"⚠️ Ошибка сплиттинга для {path.name}: {e}")
        continue

    try:
        vectordb._collection.delete(where={"source": {"$eq": str(path)}})
        vectordb.add_documents(chunks)
        print(f"✅ Проиндексировано {len(chunks)} чанков из {path.name}")
    except Exception as e:
        print(f"❌ Ошибка при заливке в Chroma для {path.name}: {e}")
        traceback.print_exc(limit=1)

print("\n🎉 Миграция в ChromaDB завершена.")
