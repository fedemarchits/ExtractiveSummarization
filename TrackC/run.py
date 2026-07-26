
"""CLI entry point.

Examples:

    python run.py --list

    python run.py \
        --model qwen35_4b \
        --dataset xsum

    python run.py \
        --model qwen35_4b \
        --dataset cnndm

Configuration is read from configs/*.yaml.

Benchmark runs are resume-safe: if a variant JSONL file already exists,
that variant is skipped.

The --list command is offline and does not require model dependencies.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXPERIMENT_CONFIGS = {
    "xsum": "configs/experiment_xsum.yaml",
    "cnndm": "configs/experiment_cnndm.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    """Create and return the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Track C extractive summarization experiments."
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Print all resolved prompt variants and exit.",
    )

    parser.add_argument(
        "--model",
        help="Model alias defined in configs/models.yaml.",
    )

    parser.add_argument(
        "--dataset",
        choices=sorted(EXPERIMENT_CONFIGS),
        help=(
            "Dataset configuration to use. Required when --model is supplied. "
            "Choices: xsum or cnndm."
        ),
    )

    parser.add_argument(
        "--models",
        default="configs/models.yaml",
        help="Path to the model configuration YAML.",
    )

    parser.add_argument(
        "--grid",
        default="configs/grid.yaml",
        help="Path to the prompt-variant grid YAML.",
    )

    return parser


def validate_file(path: str, label: str) -> None:
    """Raise a clear error when a required configuration file is missing."""
    resolved = Path(path)

    if not resolved.exists():
        raise FileNotFoundError(
            f"{label} file does not exist: {resolved}"
        )

    if not resolved.is_file():
        raise ValueError(
            f"{label} path is not a file: {resolved}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Offline grid inspection.
    if args.list:
        validate_file(args.grid, "Grid configuration")

        from engine.grid import resolve_variants

        variants = resolve_variants(args.grid)

        for variant in sorted(
            variants,
            key=lambda item: item.slug,
        ):
            print(variant.slug)

        technique_count = len(
            {variant.technique for variant in variants}
        )

        print(
            f"\n{len(variants)} variants across "
            f"{technique_count} techniques"
        )
        return

    # A benchmark requires both a model and a dataset.
    if not args.model:
        parser.error(
            "--model is required unless --list is used."
        )

    if not args.dataset:
        parser.error(
            "--dataset is required when --model is supplied."
        )

    experiment_path = EXPERIMENT_CONFIGS[args.dataset]

    validate_file(
        experiment_path,
        "Experiment configuration",
    )
    validate_file(
        args.models,
        "Model configuration",
    )
    validate_file(
        args.grid,
        "Grid configuration",
    )

    from engine.runner import run

    run(
        model_alias=args.model,
        experiment_yaml=experiment_path,
        models_yaml=args.models,
        grid_yaml=args.grid,
    )


if __name__ == "__main__":
    main()
