"""Build deterministic one-shot exemplars from Track C silver datasets.

Track C uses whole-document extractive summaries rather than ACLSum aspects.

Expected document fields:
    doc["source_sentences"] -> list[str] or serialized list
    doc["silver_indices"]   -> list[int] or serialized list of 1-based indices

Used by:
- engine.runner for answer-only one-shot techniques;
- rationale-generation scripts that try several exemplars until one is suitable.

The ``aspect``/``task`` argument is retained only for compatibility with the
Track A prompt interface. Track C should pass ``"summary"``.
"""

from __future__ import annotations

import ast
import json
import random
from typing import Iterator, List, Mapping, Optional, Sequence


from .base import Exemplar


def _get_doc(docs, index: int):
    """Read one row from a pandas DataFrame or an ordinary sequence."""
    if hasattr(docs, "iloc"):
        return docs.iloc[index]

    return docs[index]


def _parse_list(value) -> list:
    """Convert list-like or serialized list values into a Python list."""
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass

    if isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []

        return parsed if isinstance(parsed, list) else []

    return []




def _reference_summary(doc: Mapping) -> str:
    value = doc.get("reference_summary", "")
    return "" if value is None else str(value).strip()

def near_miss_indices_1based(
    doc: Mapping,
    selected: Sequence[int],
    limit: int = 5,
) -> List[int]:
    """Top non-silver sentences by ROUGE-L overlap with the human reference.

    These are hard negatives for the negative-aware prompt only; they are not
    additional silver/gold labels.
    """
    sentences = source_sentences(doc)
    reference = _reference_summary(doc)
    selected_set = set(int(i) for i in selected)
    if not reference:
        return [
            i for i in range(1, len(sentences) + 1)
            if i not in selected_set
        ][:limit]

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    scored = []
    for i, sentence in enumerate(sentences, start=1):
        if i in selected_set:
            continue
        score = scorer.score(reference, sentence)["rougeL"].fmeasure
        scored.append((i, float(score)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [i for i, _ in scored[:limit]]

def source_sentences(doc: Mapping) -> List[str]:
    """Return cleaned source sentences from one Track C silver document."""
    values = _parse_list(doc["source_sentences"])

    sentences = [
        str(sentence).strip()
        for sentence in values
        if str(sentence).strip()
    ]

    if not sentences:
        raise ValueError(
            "Track C exemplar has no valid source_sentences."
        )

    return sentences


def silver_indices_1based(doc: Mapping) -> List[int]:
    """Return clean, unique, valid 1-based silver sentence indices."""
    values = _parse_list(doc["silver_indices"])
    sentences = source_sentences(doc)
    n_sentences = len(sentences)

    cleaned: List[int] = []
    seen = set()

    for value in values:
        if isinstance(value, bool):
            continue

        try:
            index = int(value)
        except (TypeError, ValueError, OverflowError):
            continue

        if (
            1 <= index <= n_sentences
            and index not in seen
        ):
            cleaned.append(index)
            seen.add(index)

    return sorted(cleaned)


def has_gold(
    doc: Mapping,
    task: str = "summary",
) -> bool:
    """Return whether the document has at least one valid silver index."""
    del task
    return bool(silver_indices_1based(doc))


def to_exemplar(
    doc: Mapping,
    task: str = "summary",
) -> Exemplar:
    """Convert one Track C silver document into an Exemplar."""
    del task

    indices = silver_indices_1based(doc)

    if not indices:
        raise ValueError(
            "Cannot build an exemplar from a document with no silver indices."
        )

    return Exemplar(
        sentences=source_sentences(doc),
        gold_indices=indices,
        near_miss_indices=near_miss_indices_1based(doc, indices),
    )


def _shuffled_indices(
    docs,
    seed: int,
) -> List[int]:
    """Return deterministic shuffled row indices."""
    order = list(range(len(docs)))
    random.Random(seed).shuffle(order)
    return order


def select_exemplar(
    docs,
    task: str = "summary",
    seed: int = 42,
) -> Optional[Exemplar]:
    """Select one deterministic valid Track C exemplar.

    The first valid document in seeded shuffled order is returned.
    ``None`` is returned only when the input collection is empty.
    """
    if docs is None or len(docs) == 0:
        return None

    for index in _shuffled_indices(
        docs,
        seed,
    ):
        doc = _get_doc(docs, index)

        try:
            if has_gold(doc, task):
                return to_exemplar(doc, task)
        except (KeyError, TypeError, ValueError):
            continue

    raise ValueError(
        "No valid Track C exemplar was found. Every candidate was missing "
        "source_sentences, silver_indices, or contained invalid values."
    )


def iter_exemplars(
    docs,
    task: str = "summary",
    seed: int = 42,
    limit: Optional[int] = None,
) -> Iterator[Exemplar]:
    """Yield valid Track C exemplars in deterministic shuffled order."""
    if docs is None or len(docs) == 0:
        return

    if limit is not None and limit <= 0:
        return

    yielded = 0

    for index in _shuffled_indices(
        docs,
        seed,
    ):
        doc = _get_doc(docs, index)

        try:
            exemplar = to_exemplar(
                doc,
                task,
            )
        except (KeyError, TypeError, ValueError):
            continue

        yield exemplar
        yielded += 1

        if (
            limit is not None
            and yielded >= limit
        ):
            return