"""Generate Tables 10 and 11 from saved Track C predictions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io import load_all_results


TOOL_VARIANTS = [
    "tool_augmented",
    "tool_no_meta",
    "tool_no_roleplay",
]

SCORING_VARIANTS = [
    "scoring_based",
    "scoring_no_length",
    "scoring_no_redundancy",
]

DISPLAY_NAMES = {
    "tool_augmented": "Tool-Augmented Simulation (Full)",
    "tool_no_meta": "w/o Context Metadata Vectors",
    "tool_no_roleplay": "w/o Role-Play Scaffolding",
    "scoring_based": "Scoring-Based Framework (Full)",
    "scoring_no_length": "w/o Soft Length Constraints",
    "scoring_no_redundancy": "w/o Iterative Anti-Redundancy",
}


def _add_prediction_length(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "pred_length" not in result.columns:
        result["pred_length"] = result[
            "pred_indices"
        ].apply(
            lambda value: len(value)
            if isinstance(value, list)
            else 0
        )

    return result


def build_table10(
    df: pd.DataFrame,
    dataset: str = "xsum",
    model: str = "qwen35_4b",
    shot: str = "zero_shot",
    cap: str = "capped",
) -> pd.DataFrame:
    subset = df[
        (df["dataset"] == dataset)
        & (df["model"] == model)
        & (df["shot"] == shot)
        & (df["cap"] == cap)
        & (df["technique"].isin(TOOL_VARIANTS))
    ].copy()

    output = (
        subset.groupby(
            "technique",
            as_index=False,
        )
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            n_docs=("doc_id", "nunique"),
        )
    )

    output["configuration"] = output[
        "technique"
    ].map(DISPLAY_NAMES)

    return output[
        [
            "configuration",
            "technique",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
            "n_docs",
        ]
    ]


def build_table11(
    df: pd.DataFrame,
    dataset: str = "cnndm",
    model: str = "qwen35_4b",
    shot: str = "zero_shot",
    cap: str = "uncapped",
) -> pd.DataFrame:
    subset = df[
        (df["dataset"] == dataset)
        & (df["model"] == model)
        & (df["shot"] == shot)
        & (df["cap"] == cap)
        & (df["technique"].isin(SCORING_VARIANTS))
    ].copy()

    subset = _add_prediction_length(subset)

    output = (
        subset.groupby(
            "technique",
            as_index=False,
        )
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            mean_length=("pred_length", "mean"),
            n_docs=("doc_id", "nunique"),
        )
    )

    baseline_rows = output[
        output["technique"] == "scoring_based"
    ]

    if baseline_rows.empty:
        raise RuntimeError(
            "Table 11 requires the scoring_based full baseline."
        )

    baseline_length = float(
        baseline_rows.iloc[0]["mean_length"]
    )

    if baseline_length <= 0:
        raise RuntimeError(
            "Cannot calculate length delta from a zero-length baseline."
        )

    output["length_delta_pct"] = (
        (
            output["mean_length"]
            - baseline_length
        )
        / baseline_length
        * 100.0
    )

    output["configuration"] = output[
        "technique"
    ].map(DISPLAY_NAMES)

    return output[
        [
            "configuration",
            "technique",
            "precision",
            "recall",
            "f1",
            "mean_length",
            "length_delta_pct",
            "n_docs",
        ]
    ]


def build_ablation_tables(
    results_dir: str = "results",
    output_dir: str = "tables",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_all_results(results_dir)

    table10 = build_table10(df)
    table11 = build_table11(df)

    output_root = Path(output_dir)
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    table10.to_csv(
        output_root / "table10_tool_ablation.csv",
        index=False,
    )

    table11.to_csv(
        output_root / "table11_scoring_ablation.csv",
        index=False,
    )

    return table10, table11