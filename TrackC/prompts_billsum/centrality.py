"""TF-IDF cosine graph centrality, cached per document.

Ported from the old pipeline. Used by tool_augmented to annotate sentences
with a normalized [0,1] centrality score.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

_cache: Dict[Tuple[str, ...], List[float]] = {}


def centrality_scores(sentences: Sequence[str]) -> List[float]:
    key = tuple(sentences)
    if key in _cache:
        return _cache[key]

    if len(sentences) <= 1:
        _cache[key] = [1.0] * len(sentences)
        return _cache[key]

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    tfidf = TfidfVectorizer().fit_transform(list(sentences))
    scores = cosine_similarity(tfidf).sum(axis=1)
    lo, hi = scores.min(), scores.max()
    norm = (scores - lo) / (hi - lo) if hi > lo else np.ones_like(scores)
    result = norm.tolist()
    _cache[key] = result
    return result


def numbered_with_centrality(sentences: Sequence[str]) -> str:
    sc = centrality_scores(sentences)
    return "\n".join(
        f"- Sentence {i+1}: {s} [centrality: {sc[i]:.2f}]"
        for i, s in enumerate(sentences)
    )
