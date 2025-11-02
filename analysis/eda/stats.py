from __future__ import annotations

import statistics
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from .config import TFIDF_MAX_FEATURES
from .text import naive_percentile

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    TfidfVectorizer = None


def describe(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    data = list(values)
    stats_result: Dict[str, float] = {
        "count": float(len(data)),
        "min": float(min(data)),
        "max": float(max(data)),
        "mean": float(statistics.mean(data)),
        "median": float(statistics.median(data)),
    }
    percentiles = [10, 25, 50, 75, 90, 95]
    if np is not None:
        arr = np.array(data, dtype=float)
        for percentile in percentiles:
            stats_result[f"p{percentile}"] = float(np.percentile(arr, percentile))
        return stats_result

    ordered = sorted(data)
    size = len(ordered)
    for percentile in percentiles:
        stats_result[f"p{percentile}"] = naive_percentile(ordered, percentile / 100.0)
    return stats_result


def build_tfidf_report(documents: Dict[str, str], top_n: int) -> List[Tuple[str, float]]:
    if not documents or TfidfVectorizer is None or np is None:
        return []
    try:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            token_pattern=r"(?u)\b\w[\w\-]{2,}\b",
            min_df=2,
        )
        matrix = vectorizer.fit_transform(documents.values())
    except ValueError:
        return []
    if matrix.shape[1] == 0:
        return []
    mean_scores = matrix.mean(axis=0).A1
    indices = np.argsort(mean_scores)[::-1][:top_n]
    features = vectorizer.get_feature_names_out()
    return [(features[idx], float(mean_scores[idx])) for idx in indices]


def compute_term_coverage(target_terms: Sequence[str], vocabulary: Counter) -> Dict[str, object]:
    if not target_terms:
        return {}
    normalized_terms = [term.lower().strip() for term in target_terms if term.strip()]
    vocab_set = set(vocabulary.keys())
    present = sorted(term for term in normalized_terms if term in vocab_set)
    missing = sorted(set(normalized_terms) - set(present))
    coverage = len(present) / len(normalized_terms) if normalized_terms else 0.0
    per_term = {term: term in vocab_set for term in normalized_terms}
    return {
        "coverage": coverage,
        "present": present,
        "missing": missing,
        "per_term": per_term,
    }
