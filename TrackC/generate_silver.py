"""Command-line entry point for Track C silver-dataset generation.

Examples:

    python generate_silver.py --dataset xsum

    python generate_silver.py --dataset cnndm

    python generate_silver.py \
        --dataset xsum \
        --split validation \
        --out data/silver/xsum_validation_silver.parquet

    python generate_silver.py \
        --dataset cnndm \
        --max-docs 100

The actual heuristic implementation lives in engine/silver.py.
This script only validates arguments, resolves defaults, and starts generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from engine.silver import generate_silver_dataset


DEFAULT_OUTPUTS = {
    ("xsum", "test"): "data/silver/xsum_silver.parquet",
    ("xsum", "validation"): "data/silver/xsum_validation_silver.parquet",
    ("cnndm", "test"): "data/silver/cnndm_silver.parquet",
    ("cnndm", "validation"): "data/silver/cnndm_validation_silver.parquet",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Track C silver extractive datasets using the "
            "best Track B heuristic."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["xsum", "cnndm"],
        help="Abstractive dataset to convert into silver extractive labels.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help=(
            "Dataset split to convert. Test is used for evaluation; "
            "validation may be generated for one-shot exemplars."
        ),
    )

    parser.add_argument(
        "--heuristic",
        default="best_heuristic.json",
        help="Path to the Track B best-heuristic JSON file.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output parquet path. If omitted, a standard path under "
            "data/silver/ is selected automatically."
        ),
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help=(
            "Maximum number of documents to process. "
            "Omit to process the entire selected split."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "If set (with --max-docs), take a RANDOM subsample of that many "
            "documents using this seed. Omit to take the first --max-docs. "
            "Same seed + same split => same documents (reproducible, and "
            "identical across models)."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing silver parquet file.",
    )

    return parser


def validate_heuristic(path: Path) -> Dict:
    """Load and validate the Track B heuristic configuration."""
    if not path.exists():
        raise FileNotFoundError(
            f"Best-heuristic JSON does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Best-heuristic path is not a file: {path}"
        )

    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)

    required = {
        "best_heuristic",
        "best_k",
    }

    missing = required - set(config)

    if missing:
        raise ValueError(
            f"Heuristic configuration is missing fields: "
            f"{sorted(missing)}"
        )

    if config["best_heuristic"] != "local_score":
        raise ValueError(
            "This Track C generator currently supports only "
            f"'local_score', but the JSON specifies "
            f"{config['best_heuristic']!r}."
        )

    try:
        best_k = int(config["best_k"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "'best_k' must be a positive integer."
        ) from exc

    if best_k <= 0:
        raise ValueError(
            f"'best_k' must be positive, got {best_k}."
        )

    extract_metric = config.get("extract_metric", "rouge1")

    if extract_metric not in {
        "rouge1",
        "rouge2",
        "rougeL",
    }:
        raise ValueError(
            "Unsupported extract_metric in heuristic JSON: "
            f"{extract_metric!r}"
        )

    return config


def resolve_output_path(
    dataset: str,
    split: str,
    explicit_path: str | None,
) -> Path:
    """Resolve the requested output path."""
    if explicit_path:
        return Path(explicit_path)

    default = DEFAULT_OUTPUTS.get((dataset, split))

    if default is not None:
        return Path(default)

    return Path(
        f"data/silver/{dataset}_{split}_silver.parquet"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_docs is not None and args.max_docs <= 0:
        parser.error("--max-docs must be greater than zero.")

    heuristic_path = Path(args.heuristic)
    heuristic = validate_heuristic(heuristic_path)

    output_path = resolve_output_path(
        dataset=args.dataset,
        split=args.split,
        explicit_path=args.out,
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("[silver] Track C silver generation")
    print(f"[silver] dataset: {args.dataset}")
    print(f"[silver] split: {args.split}")
    print(
        f"[silver] heuristic: "
        f"{heuristic['best_heuristic']}"
    )
    print(
        f"[silver] extract metric: "
        f"{heuristic.get('extract_metric', 'rouge1')}"
    )
    print(f"[silver] best K: {heuristic['best_k']}")
    print(f"[silver] output: {output_path}")

    if args.seed is not None:
        print(f"[silver] random subsample seed: {args.seed}")

    dataframe = generate_silver_dataset(
        dataset_name=args.dataset,
        split=args.split,
        heuristic_path=heuristic_path,
        out_path=output_path,
        max_docs=args.max_docs,
        seed=args.seed,
    )

    print(
        f"[silver] complete: wrote {len(dataframe)} documents "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()

