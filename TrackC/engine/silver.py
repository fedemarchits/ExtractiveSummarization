
"""Track C silver-dataset generation.

This module converts abstractive summarization datasets into silver extractive
datasets using the best heuristic selected by Track B.

Supported datasets:
- XSum
- CNN/DailyMail

Current supported heuristic:
- local_score

Expected heuristic configuration:
{
    "best_heuristic": "local_score",
    "best_k": 4,
    "extract_metric": "rouge1",
    "mode": "singular"
}
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union


import pandas as pd
from datasets import Dataset, load_dataset
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize



PathLike = Union[str, Path]

_SUPPORTED_METRICS = {
    "rouge1",
    "rouge2",
    "rougeL",
}

_SUPPORTED_DATASETS = {
    "xsum",
    "cnndm",
    "cnn_dailymail",
    "cnn/dailymail",
}

# Reuse one scorer instead of constructing it for every sentence.
_ROUGE_SCORER = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
)


def load_best_heuristic(path: PathLike) -> Dict[str, Any]:
    """Load and validate the Track B best-heuristic configuration."""
    heuristic_path = Path(path)

    if not heuristic_path.exists():
        raise FileNotFoundError(
            f"Best-heuristic file does not exist: {heuristic_path}"
        )

    if not heuristic_path.is_file():
        raise ValueError(
            f"Best-heuristic path is not a file: {heuristic_path}"
        )

    try:
        with heuristic_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in heuristic file: {heuristic_path}"
        ) from exc

    required = {
        "best_heuristic",
        "best_k",
    }

    missing = required - set(config)

    if missing:
        raise ValueError(
            "Heuristic configuration is missing required fields: "
            f"{sorted(missing)}"
        )

    heuristic_name = str(config["best_heuristic"]).strip()

    if heuristic_name != "local_score":
        raise ValueError(
            "This Track C implementation currently supports only "
            f"'local_score', but received {heuristic_name!r}."
        )

    try:
        best_k = int(config["best_k"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "'best_k' must be a positive integer."
        ) from exc

    if best_k <= 0:
        raise ValueError(
            f"'best_k' must be greater than zero, got {best_k}."
        )

    metric = str(
        config.get("extract_metric", "rouge1")
    ).strip()

    if metric not in _SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported extract_metric {metric!r}. "
            f"Supported metrics: {sorted(_SUPPORTED_METRICS)}"
        )

    config["best_k"] = best_k
    config["extract_metric"] = metric
    config["mode"] = str(
        config.get("mode", "singular")
    ).strip()

    return config

def _try_nltk_sent_tokenize(text: str) -> List[str]:
    """Sentence segmentation using NLTK Punkt."""
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass


        return [
            s.strip()
            for s in sent_tokenize(text)
            if s.strip()
        ]

    except Exception:
        return []


def _regex_sentence_split(text: str) -> List[str]:
    """Fallback deterministic sentence splitter."""
    chunks = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


def sentence_split(text: str) -> List[str]:
    """Split raw document text into sentences.

    Track B used pre-segmented ACLSum sentences.
    Track C receives raw XSum/CNN-DailyMail documents and therefore performs
    deterministic sentence segmentation before applying the unchanged
    local-score heuristic.
    """
    text = re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()

    if not text:
        return []

    sentences = _try_nltk_sent_tokenize(text)

    if sentences:
        return sentences

    return _regex_sentence_split(text)


# def sentence_split(text: str) -> List[str]:
#     """Split a document into source sentences.

#     This lightweight splitter is deterministic and dependency-free.

#     Important:
#     For strict reproducibility with Track B, replace this implementation with
#     the exact same sentence splitter used by Track B if it differs.
#     """
#     normalized = re.sub(
#         r"\s+",
#         " ",
#         str(text or ""),
#     ).strip()

#     if not normalized:
#         return []

#     parts = re.split(
#         r"(?<=[.!?])\s+",
#         normalized,
#     )

#     return [
#         part.strip()
#         for part in parts
#         if part and part.strip()
#     ]


def score_sentence(
    sentence: str,
    reference: str,
    metric: str = "rouge1",
) -> float:
    """Score one source sentence against the abstractive reference."""
    if metric not in _SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric {metric!r}. "
            f"Supported metrics: {sorted(_SUPPORTED_METRICS)}"
        )

    sentence = str(sentence or "").strip()
    reference = str(reference or "").strip()

    if not sentence or not reference:
        return 0.0

    scores = _ROUGE_SCORER.score(
        reference,
        sentence,
    )

    return float(scores[metric].fmeasure)


def local_score_select(
    sentences: Sequence[str],
    reference_summary: str,
    k: int,
    metric: str = "rouge1",
) -> List[int]:
    """Select the top-K individually scored source sentences.

    Returns 1-based sentence indices in source-document order.

    Ranking ties are resolved by source position, making results deterministic.
    """
    if k <= 0:
        raise ValueError(
            f"k must be greater than zero, got {k}."
        )

    if metric not in _SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric {metric!r}."
        )

    if not sentences:
        return []

    scored: List[Tuple[int, float]] = []

    for index, sentence in enumerate(
        sentences,
        start=1,
    ):
        score = score_sentence(
            sentence=sentence,
            reference=reference_summary,
            metric=metric,
        )

        scored.append(
            (index, score)
        )

    # Higher score first; earlier sentence wins ties.
    ranked = sorted(
        scored,
        key=lambda item: (-item[1], item[0]),
    )

    selected = [
        index
        for index, _ in ranked[: min(k, len(ranked))]
    ]

    # Reconstructive summaries should preserve source order.
    return sorted(selected)


def load_abstractive_dataset(
    dataset_name: str,
    split: str,
) -> Tuple[Dataset, str, str]:
    """Load an abstractive dataset and return its text-column names."""
    normalized_name = str(dataset_name).strip().lower()

    if normalized_name not in _SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}. "
            "Supported values: xsum or cnndm."
        )

    if normalized_name == "xsum":
        dataset = load_dataset(
            "EdinburghNLP/xsum",
            split=split,
        )

        return (
            dataset,
            "document",
            "summary",
        )

    dataset = load_dataset(
        "cnn_dailymail",
        "3.0.0",
        split=split,
    )

    return (
        dataset,
        "article",
        "highlights",
    )


def _safe_text(value: Any) -> str:
    """Convert missing or non-string dataset values into safe text."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def _document_id(
    example: Dict[str, Any],
    dataset_name: str,
    split: str,
    index: int,
) -> str:
    """Return a stable document identifier."""
    raw_id = example.get("id")

    if raw_id is not None:
        cleaned = _safe_text(raw_id)

        if cleaned:
            return cleaned

    return f"{dataset_name}_{split}_{index}"


def generate_silver_dataset(
    dataset_name: str,
    split: str,
    heuristic_path: PathLike,
    out_path: PathLike,
    max_docs: int | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate and save one Track C silver dataset.

    Parameters
    ----------
    dataset_name:
        ``xsum`` or ``cnndm``.
    split:
        Dataset split such as ``test`` or ``validation``.
    heuristic_path:
        Path to Track B's ``best_heuristic.json``.
    out_path:
        Destination parquet file.
    max_docs:
        Optional maximum number of documents.

    Returns
    -------
    pandas.DataFrame
        Generated silver dataset.
    """
    if max_docs is not None and max_docs <= 0:
        raise ValueError(
            f"max_docs must be greater than zero, got {max_docs}."
        )

    heuristic = load_best_heuristic(
        heuristic_path
    )

    heuristic_name = heuristic["best_heuristic"]
    best_k = int(heuristic["best_k"])
    metric = heuristic["extract_metric"]
    mode = heuristic["mode"]

    dataset, article_column, summary_column = (
        load_abstractive_dataset(
            dataset_name=dataset_name,
            split=split,
        )
    )

    if max_docs is not None:
        selected_count = min(
            max_docs,
            len(dataset),
        )

        # With a seed, take a RANDOM subsample (shuffle then take the first
        # selected_count); without one, keep the original first-N behavior.
        # Shuffle is an index permutation, so only the selected docs are then
        # run through the heuristic.
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)

        dataset = dataset.select(
            range(selected_count)
        )

    normalized_dataset_name = (
        "cnndm"
        if dataset_name in {
            "cnn_dailymail",
            "cnn/dailymail",
        }
        else dataset_name
    )

    rows: List[Dict[str, Any]] = []

    for index, example in enumerate(dataset):
        article = _safe_text(
            example.get(article_column)
        )

        reference_summary = _safe_text(
            example.get(summary_column)
        )

        sentences = sentence_split(article)

        if not sentences or not reference_summary:
            silver_indices: List[int] = []
            silver_text = ""
        else:
            silver_indices = local_score_select(
                sentences=sentences,
                reference_summary=reference_summary,
                k=min(best_k, len(sentences)),
                metric=metric,
            )

            silver_text = " ".join(
                sentences[sentence_index - 1]
                for sentence_index in silver_indices
                if 1 <= sentence_index <= len(sentences)
            )

        rows.append(
            {
                "id": _document_id(
                    example=example,
                    dataset_name=normalized_dataset_name,
                    split=split,
                    index=index,
                ),
                "dataset": normalized_dataset_name,
                "split": split,
                "source_text": article,
                "source_sentences": sentences,
                "reference_summary": reference_summary,
                "silver_indices": silver_indices,
                "silver_text": silver_text,
                "num_source_sentences": len(sentences),
                "num_silver_sentences": len(silver_indices),
                "heuristic": heuristic_name,
                "extract_metric": metric,
                "mode": mode,
                "best_k": best_k,
            }
        )

    dataframe = pd.DataFrame(rows)

    output_path = Path(out_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        dataframe.to_parquet(
            output_path,
            index=False,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Writing parquet files requires pyarrow or fastparquet. "
            "Install pyarrow with: pip install pyarrow"
        ) from exc

    return dataframe

