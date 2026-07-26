"""Position-bias analysis for Track C.

This module compares:

1. Sentence positions selected by each LLM prompting strategy.
2. Sentence positions selected by the Track B silver-label heuristic.

This distinction is important because an apparent model lead bias may already
exist in the silver reference labels.

Input structure:

    results/<dataset>/<model>/<variant_slug>.jsonl

Required JSONL fields:

    doc_id
    num_sentences
    pred_indices
    silver_indices

Outputs:

    tables/position_bias.csv
    figures/position_bias_<dataset>_<model>.pdf
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import load_all_results


def normalize_indices(value) -> List[int]:
    """Convert an index value into a clean unique integer list."""
    import ast
    import json

    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        try:
            value = json.loads(value)
        except Exception:
            try:
                value = ast.literal_eval(value)
            except Exception:
                return []

    if not isinstance(value, (list, tuple, set)):
        return []

    cleaned: List[int] = []
    seen = set()

    for item in value:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue

        if index > 0 and index not in seen:
            cleaned.append(index)
            seen.add(index)

    return cleaned


def normalized_positions(
    indices: Iterable[int],
    n_sentences: int,
) -> List[float]:
    """Convert 1-based indices to normalized positions in [0, 1].

    The first sentence maps to 0.0 and the final sentence maps to 1.0.
    A single-sentence document maps its only sentence to 0.0.
    """
    if n_sentences <= 0:
        return []

    clean_indices = normalize_indices(indices)

    if n_sentences == 1:
        return [
            0.0
            for index in clean_indices
            if index == 1
        ]

    return [
        (index - 1) / (n_sentences - 1)
        for index in clean_indices
        if 1 <= index <= n_sentences
    ]


def _expand_model_positions(df: pd.DataFrame) -> List[dict]:
    """Expand every model-selected sentence into one position row."""
    expanded: List[dict] = []

    for _, row in df.iterrows():
        try:
            n_sentences = int(row["num_sentences"])
        except (TypeError, ValueError):
            continue

        positions = normalized_positions(
            row.get("pred_indices", []),
            n_sentences,
        )

        for position in positions:
            expanded.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "technique": row["technique"],
                    "shot": row["shot"],
                    "cap": row["cap"],
                    "variant_slug": row.get(
                        "variant_slug",
                        "",
                    ),
                    "doc_id": row["doc_id"],
                    "selection_source": "model",
                    "normalized_position": position,
                }
            )

    return expanded


def _expand_silver_positions(df: pd.DataFrame) -> List[dict]:
    """Expand silver-label positions once per dataset/document.

    Silver labels are repeated in every model/strategy result JSONL file.
    Deduplication prevents the same silver target from being counted once for
    every strategy or model.
    """
    expanded: List[dict] = []

    required = {
        "dataset",
        "doc_id",
        "num_sentences",
        "silver_indices",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Position-bias analysis is missing required columns: "
            f"{sorted(missing)}"
        )

    silver_docs = df.drop_duplicates(
        subset=["dataset", "doc_id"],
        keep="first",
    )

    for _, row in silver_docs.iterrows():
        try:
            n_sentences = int(row["num_sentences"])
        except (TypeError, ValueError):
            continue

        positions = normalized_positions(
            row.get("silver_indices", []),
            n_sentences,
        )

        for position in positions:
            expanded.append(
                {
                    "dataset": row["dataset"],
                    "model": "silver_reference",
                    "technique": "silver_reference",
                    "shot": "reference",
                    "cap": "reference",
                    "variant_slug": "silver_reference",
                    "doc_id": row["doc_id"],
                    "selection_source": "silver",
                    "normalized_position": position,
                }
            )

    return expanded


def build_position_table(
    results_dir: str = "results",
    n_bins: int = 10,
    out_path: str = "tables/position_bias.csv",
) -> pd.DataFrame:
    """Build model and silver position distributions.

    The output has one row per:

        dataset × model × technique × shot × cap
        × selection_source × position_bin
    """
    if n_bins <= 0:
        raise ValueError(
            f"n_bins must be greater than zero, got {n_bins}."
        )

    df = load_all_results(results_dir)

    if df.empty:
        return pd.DataFrame()

    required_model_columns = {
        "dataset",
        "model",
        "technique",
        "shot",
        "cap",
        "doc_id",
        "num_sentences",
        "pred_indices",
        "silver_indices",
    }

    missing = required_model_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Position-bias analysis cannot run because result data is "
            f"missing columns: {sorted(missing)}"
        )

    expanded = [
        *_expand_model_positions(df),
        *_expand_silver_positions(df),
    ]

    selected = pd.DataFrame(expanded)

    if selected.empty:
        return selected

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    selected["position_bin"] = pd.cut(
        selected["normalized_position"],
        bins=bin_edges,
        include_lowest=True,
        labels=False,
    )

    selected = selected.dropna(
        subset=["position_bin"]
    ).copy()

    selected["position_bin"] = (
        selected["position_bin"].astype(int)
    )

    group_columns = [
        "dataset",
        "model",
        "technique",
        "shot",
        "cap",
        "variant_slug",
        "selection_source",
        "position_bin",
    ]

    summary = (
        selected.groupby(
            group_columns,
            as_index=False,
            observed=True,
        )
        .size()
        .rename(
            columns={
                "size": "selection_count",
            }
        )
    )

    # Add zero-count rows for missing bins so every curve spans the same axis.
    group_identity = [
        "dataset",
        "model",
        "technique",
        "shot",
        "cap",
        "variant_slug",
        "selection_source",
    ]

    completed_groups: List[pd.DataFrame] = []

    for keys, group in summary.groupby(
        group_identity,
        dropna=False,
    ):
        full_bins = pd.DataFrame(
            {
                "position_bin": range(n_bins),
            }
        )

        for column, value in zip(
            group_identity,
            keys if isinstance(keys, tuple) else (keys,),
        ):
            full_bins[column] = value

        completed = full_bins.merge(
            group,
            on=group_identity + ["position_bin"],
            how="left",
        )

        completed["selection_count"] = (
            completed["selection_count"]
            .fillna(0)
            .astype(int)
        )

        completed_groups.append(completed)

    summary = pd.concat(
        completed_groups,
        ignore_index=True,
    )

    totals = summary.groupby(
        group_identity
    )["selection_count"].transform("sum")

    summary["selection_probability"] = np.where(
        totals > 0,
        summary["selection_count"] / totals,
        0.0,
    )

    summary["bin_start"] = (
        summary["position_bin"] / n_bins
    )

    summary["bin_end"] = (
        (summary["position_bin"] + 1) / n_bins
    )

    summary = summary.sort_values(
        group_identity + ["position_bin"]
    ).reset_index(drop=True)

    output_path = Path(out_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    return summary


def _silver_curve_for_dataset(
    table: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    """Return the shared silver-reference position curve."""
    silver = table[
        (table["dataset"] == dataset)
        & (table["selection_source"] == "silver")
    ].copy()

    if silver.empty:
        return silver

    # There should already be one silver row set per dataset, but aggregation
    # makes this robust to externally produced or older position tables.
    return (
        silver.groupby(
            "position_bin",
            as_index=False,
        )
        .agg(
            selection_count=(
                "selection_count",
                "sum",
            ),
        )
        .assign(
            selection_probability=lambda frame: (
                frame["selection_count"]
                / frame["selection_count"].sum()
                if frame["selection_count"].sum() > 0
                else 0.0
            )
        )
        .sort_values("position_bin")
    )


def plot_position_bias(
    table: pd.DataFrame,
    dataset: str,
    model: str,
    out_path: str,
) -> None:
    """Plot prompting-strategy curves and the shared silver reference."""
    model_subset = table[
        (table["dataset"] == dataset)
        & (table["model"] == model)
        & (table["selection_source"] == "model")
    ].copy()

    if model_subset.empty:
        print(
            "[warning] No model position-bias data found for "
            f"dataset={dataset!r}, model={model!r}."
        )
        return

    silver_curve = _silver_curve_for_dataset(
        table,
        dataset,
    )

    n_bins = int(
        table["position_bin"].max()
    ) + 1

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    variant_group_columns = [
        "technique",
        "shot",
        "cap",
        "variant_slug",
    ]

    for keys, group in model_subset.groupby(
        variant_group_columns,
        dropna=False,
    ):
        technique, shot, cap, variant_slug = keys

        group = group.sort_values(
            "position_bin"
        )

        label = (
            variant_slug
            if variant_slug
            else f"{technique} / {shot} / {cap}"
        )

        axis.plot(
            group["position_bin"],
            group["selection_probability"],
            marker="o",
            label=label,
        )

    if not silver_curve.empty:
        axis.plot(
            silver_curve["position_bin"],
            silver_curve["selection_probability"],
            marker="s",
            linestyle="--",
            linewidth=2.5,
            label="silver reference",
        )

    tick_positions = np.arange(n_bins)

    tick_labels = [
        (
            f"{int(100 * bin_index / n_bins)}–"
            f"{int(100 * (bin_index + 1) / n_bins)}%"
        )
        for bin_index in range(n_bins)
    ]

    axis.set_xticks(
        tick_positions
    )

    axis.set_xticklabels(
        tick_labels,
        rotation=35,
        ha="right",
    )

    axis.set_xlabel(
        "Normalized document depth"
    )

    axis.set_ylabel(
        "Selection probability"
    )

    axis.set_title(
        f"Position bias — {dataset} — {model}"
    )

    axis.set_ylim(bottom=0.0)
    axis.legend(
        fontsize="small",
    )

    figure.tight_layout()

    output_path = Path(out_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """Build the table and generate one figure per dataset/model."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compare LLM and silver-reference sentence-position "
            "distributions."
        )
    )

    parser.add_argument(
        "--results",
        default="results",
    )

    parser.add_argument(
        "--table-out",
        default="tables/position_bias.csv",
    )

    parser.add_argument(
        "--figures-dir",
        default="figures",
    )

    parser.add_argument(
        "--n-bins",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    table = build_position_table(
        results_dir=args.results,
        n_bins=args.n_bins,
        out_path=args.table_out,
    )

    if table.empty:
        print(
            "[warning] No position-bias data was generated."
        )
        return

    model_rows = table[
        table["selection_source"] == "model"
    ][["dataset", "model"]].drop_duplicates()

    figures_dir = Path(args.figures_dir)

    for row in model_rows.itertuples(index=False):
        plot_position_bias(
            table=table,
            dataset=row.dataset,
            model=row.model,
            out_path=str(
                figures_dir
                / f"position_bias_{row.dataset}_{row.model}.pdf"
            ),
        )

    print(
        f"[written] {args.table_out}"
    )


if __name__ == "__main__":
    main()