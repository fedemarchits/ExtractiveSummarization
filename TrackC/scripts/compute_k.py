"""Derive the sentence budget K for a dataset from its compression ratio.

K is the number of source sentences a reference-length extractive summary
needs. It is estimated on a development split, never on the evaluation split:

    compression ratio  CR_d = reference-summary words / source-document words
    per-document K     K_d  = CR_d * number of source sentences
    dataset K               = round(median over documents of K_d), at least 1

The median is used because both document length and summary length are
long-tailed. BillSum ships no validation split, so its train split is the
development set (the test split stays untouched).

The result is written as a heuristic JSON in the same format as Track B's
``best_heuristic.json``, so ``generate_silver.py`` and the runner consume it
unchanged.

Usage:

    python -m scripts.compute_k --dataset billsum

    python -m scripts.compute_k \
        --dataset billsum \
        --split train \
        --max-docs 1000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

# Development split used to estimate K (never the evaluation split).
DEV_SPLITS = {
    "xsum": "validation",
    "cnndm": "validation",
    "billsum": "train",
}

DEFAULT_OUTPUTS = {
    "billsum": "best_heuristic_billsum.json",
}


def _percentile(values: List[float], q: float) -> float:
    import numpy as np

    return float(np.percentile(values, q))


def compute_k(
    dataset_name: str,
    split: str,
    max_docs: int | None = None,
    seed: int | None = None,
) -> Dict:
    """Return compression-ratio statistics and the derived K."""
    import numpy as np

    from engine.silver import load_abstractive_dataset, sentence_split

    dataset, article_column, summary_column = load_abstractive_dataset(
        dataset_name=dataset_name,
        split=split,
    )

    if max_docs is not None:
        if seed is not None:
            dataset = dataset.shuffle(seed=seed)

        dataset = dataset.select(
            range(min(max_docs, len(dataset)))
        )

    ratios: List[float] = []
    sentence_counts: List[int] = []
    per_doc_k: List[float] = []

    for example in dataset:
        article = str(example.get(article_column) or "")
        summary = str(example.get(summary_column) or "")

        doc_words = len(article.split())
        summary_words = len(summary.split())

        sentences = sentence_split(
            article,
            dataset_name=dataset_name,
        )

        if not doc_words or not summary_words or not sentences:
            continue

        ratio = summary_words / doc_words

        ratios.append(ratio)
        sentence_counts.append(len(sentences))
        per_doc_k.append(ratio * len(sentences))

    if not per_doc_k:
        raise RuntimeError(
            f"No usable documents in {dataset_name}/{split}."
        )

    best_k = max(1, int(round(float(np.median(per_doc_k)))))

    return {
        "best_k": best_k,
        "n_docs": len(per_doc_k),
        "compression_ratio_mean": round(float(np.mean(ratios)), 4),
        "compression_ratio_median": round(float(np.median(ratios)), 4),
        "sentences_per_doc_mean": round(float(np.mean(sentence_counts)), 2),
        "sentences_per_doc_median": float(np.median(sentence_counts)),
        "k_per_doc_mean": round(float(np.mean(per_doc_k)), 3),
        "k_per_doc_median": round(float(np.median(per_doc_k)), 3),
        "k_per_doc_p25": round(_percentile(per_doc_k, 25), 3),
        "k_per_doc_p75": round(_percentile(per_doc_k, 75), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive K from the compression ratio of a development split."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DEV_SPLITS),
    )

    parser.add_argument(
        "--split",
        default=None,
        help="Development split. Defaults to the dataset's dev split.",
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional subsample size. Omit to use the whole split.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for a random subsample (with --max-docs).",
    )

    parser.add_argument(
        "--base-heuristic",
        default="best_heuristic.json",
        help="Track B heuristic JSON whose non-K settings are inherited.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output heuristic JSON. Defaults to best_heuristic_<dataset>.json "
            "for BillSum; required for other datasets so Track B's file is "
            "never overwritten by accident."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the statistics without writing the JSON.",
    )

    args = parser.parse_args()

    split = args.split or DEV_SPLITS[args.dataset]
    out = args.out or DEFAULT_OUTPUTS.get(args.dataset)

    if out is None and not args.dry_run:
        parser.error(
            f"--out is required for --dataset {args.dataset} "
            "(or pass --dry-run)."
        )

    stats = compute_k(
        dataset_name=args.dataset,
        split=split,
        max_docs=args.max_docs,
        seed=args.seed,
    )

    print(f"[k] dataset: {args.dataset}  split: {split}")

    for key, value in stats.items():
        print(f"[k] {key}: {value}")

    if args.dry_run:
        return

    with Path(args.base_heuristic).open("r", encoding="utf-8") as fh:
        base = json.load(fh)

    heuristic = {
        "best_heuristic": base["best_heuristic"],
        "best_k": stats["best_k"],
        "extract_metric": base.get("extract_metric", "rouge1"),
        "mode": base.get("mode", "singular"),
        "k_selection": {
            "method": "compression_ratio",
            "dataset": args.dataset,
            "split": split,
            "max_docs": args.max_docs,
            "seed": args.seed,
            **{
                key: value
                for key, value in stats.items()
                if key != "best_k"
            },
        },
    }

    out_path = Path(out)

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(heuristic, fh, indent=2)
        fh.write("\n")

    print(f"[k] wrote {out_path} (best_k = {stats['best_k']})")


if __name__ == "__main__":
    main()
