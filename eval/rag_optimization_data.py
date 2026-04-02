"""
Данные для ноутбука rag_optimization_report.ipynb
Этот файл содержит все метрики в виде Python-словарей
"""

EMBEDDER_COMPARISON = {
    "baseline": {
        "collection": "default",
        "embed_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "chunking": {"chunk_size": 650, "chunk_overlap": 250},
        "retrieval": {
            "recall@5": 0.0099,
            "recall@10": 0.0198,
            "precision@5": 0.0059,
            "precision@10": 0.0040,
            "mrr": 0.0142,
            "ndcg@3": 0.0091
        },
        "answer": {
            "avg_ref_sim": 0.5897,
            "avg_refs_sim": 0.5958
        },
        "notes": "Original production config: mpnet embedder + RecursiveCharacterTextSplitter(650, 250)"
    },
    "FRIDA": {
        "collection": "test_frida",
        "chunking": {"chunk_size": 650, "chunk_overlap": 250},  # Те же чанки, другой эмбеддер
        "retrieval": {
            "recall@5": 0.0644,
            "recall@10": 0.0644,
            "precision@5": 0.0277,
            "precision@10": 0.0228,
            "mrr": 0.0436,
            "ndcg@3": 0.0401
        },
        "answer": {
            "avg_ref_sim": 0.6194,
            "avg_refs_sim": 0.6247
        },
        "notes": "T5 encoder, works on GPU. Те же чанки 650/250, просто замена эмбеддера"
    },
    "bge-m3": {
        "collection": "test_bge_m3",
        "chunking": {"chunk_size": 650, "chunk_overlap": 250},  # Те же чанки, другой эмбеддер
        "retrieval": {
            "recall@5": 0.5941,
            "recall@10": 0.6436,
            "precision@5": 0.1802,
            "precision@10": 0.1050,
            "mrr": 0.4734,
            "ndcg@3": 0.5507
        },
        "answer": {
            "avg_ref_sim": 0.6221,
            "avg_refs_sim": 0.6271
        },
        "notes": "Multi-lingual, 8192 tokens, works on GPU (RTX 5080). Те же чанки 650/250, просто замена эмбеддера"
    }
}

CHUNKING_COMPARISON = {
    "1000/200": {
        "recall@5": 0.3515,
        "recall@10": 0.4356,
        "precision@5": 0.1525,
        "precision@10": 0.1267,
        "mrr": 0.2940,
        "ndcg@3": 0.4000,
        "total_chunks": 61568,
        "notes": "Альтернативная конфигурация: перечанковывание оригиналов с 1000/200. Best among alternative chunking"
    },
    "400/100": {
        "recall@5": 0.2970,
        "recall@10": 0.3812,
        "precision@5": 0.1267,
        "precision@10": 0.1109,
        "mrr": 0.0506,
        "ndcg@3": 0.0476,
        "total_chunks": 110588,
        "notes": "Альтернативная конфигурация: перечанковывание оригиналов с 400/100. Worst metrics"
    },
    "800/150": {
        "recall@5": 0.3218,
        "recall@10": 0.3960,
        "precision@5": 0.1366,
        "precision@10": 0.1208,
        "mrr": 0.0581,
        "ndcg@3": 0.0487,
        "total_chunks": 49838,
        "notes": "Альтернативная конфигурация: перечанковывание оригиналов с 800/150. Balanced config"
    },
    "Без чанкования (базовые 650/250)": {
        "recall@5": 0.5941,
        "recall@10": 0.6436,
        "precision@5": 0.1802,
        "precision@10": 0.1050,
        "mrr": 0.4734,
        "ndcg@3": 0.5507,
        "total_docs": 61543,
        "notes": "Оригинальная продакшен конфигурация: 650/250 из ChromaDB. Best overall"
    }
}

# Сводные данные для финального сравнения
SUMMARY_DATA = {
    "best_config": {
        "embedder": "BAAI/bge-m3",
        "chunking": None,
        "collection": "test_bge_m3",
        "recall@10": 0.6436,
        "mrr": 0.4734,
        "avg_ref_sim": 0.6221,
        "improvement_vs_baseline": "+3150%"
    },
    "best_chunked": {
        "embedder": "BAAI/bge-m3",
        "chunking": "1000/200",
        "collection": "test_chunk_1000_200",
        "recall@10": 0.4356,
        "mrr": 0.2940,
        "avg_ref_sim": 0.62,
        "improvement_vs_baseline": "+2100%"
    }
}


if __name__ == "__main__":
    print_summary()
