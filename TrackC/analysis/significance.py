"""Paired significance tests between Track C prompting strategies.

For each statistical family defined by:

    dataset × model × shot × cap

the module:

1. pairs strategies on the same document IDs;
2. applies a two-sided Wilcoxon signed-rank test;
3. adjusts p-values with Benjamini-Hochberg only within that family;
4. saves the resulting comparison table.

Keeping correction within each family avoids mixing unrelated comparisons from
different datasets, models, shot settings, or cap settings.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .io import load_all_results


FAMILY_COLUMNS = [
    "dataset",
    "model",
    "shot",
    "cap",
]


def benjamini_hochberg(
    p_values: Iterable[float],
) -> List[float]:
    """Return Benjamini-Hochberg adjusted p-values.

    The correction is applied to one predefined statistical family at a time.
    NaN p-values remain NaN.
    """
    values = np.asarray(
        list(p_values),
        dtype=float,
    )

    adjusted = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    valid_mask = np.isfinite(values)
    valid_indices = np.where(valid_mask)[0]
    valid_values = values[valid_mask]

    if len(valid_values) == 0:
        return adjusted.tolist()

    order = np.argsort(valid_values)
    ordered_p = valid_values[order]

    n_tests = len(ordered_p)

    ordered_adjusted = np.empty(
        n_tests,
        dtype=float,
    )

    running_minimum = 1.0

    for reverse_position in range(
        n_tests - 1,
        -1,
        -1,
    ):
        rank = reverse_position + 1

        corrected = (
            ordered_p[reverse_position]
            * n_tests
            / rank
        )

        running_minimum = min(
            running_minimum,
            corrected,
        )

        ordered_adjusted[reverse_position] = min(
            running_minimum,
            1.0,
        )

    restored = np.empty(
        n_tests,
        dtype=float,
    )

    restored[order] = ordered_adjusted
    adjusted[valid_indices] = restored

    return adjusted.tolist()


def _prepare_strategy_scores(
    group: pd.DataFrame,
    technique: str,
    metric: str,
) -> pd.DataFrame:
    """Return one metric value per document for one technique."""
    subset = group[
        group["technique"] == technique
    ][["doc_id", metric]].copy()

    subset[metric] = pd.to_numeric(
        subset[metric],
        errors="coerce",
    )

    # A strategy should normally have one row per document. If duplicate rows
    # exist, average them so the paired test still has one value per doc_id.
    return (
        subset.groupby(
            "doc_id",
            as_index=False,
        )[metric]
        .mean()
    )


def _paired_wilcoxon(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float]:
    """Run a robust paired two-sided Wilcoxon test."""
    differences = (
        x.to_numpy(dtype=float)
        - y.to_numpy(dtype=float)
    )

    # scipy raises when every paired difference is zero under zero_method
    # "wilcox". In that case, there is clearly no evidence of a difference.
    if len(differences) == 0:
        return float("nan"), float("nan")

    if np.allclose(
        differences,
        0.0,
        equal_nan=False,
    ):
        return 0.0, 1.0

    try:
        result = wilcoxon(
            x,
            y,
            alternative="two-sided",
            zero_method="wilcox",
        )

        return (
            float(result.statistic),
            float(result.pvalue),
        )

    except ValueError:
        return float("nan"), float("nan")


def paired_strategy_tests(
    results_dir: str = "results",
    metric: str = "f1",
    out_path: str = "tables/significance_f1.csv",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare all prompting strategies with paired Wilcoxon tests.

    Multiple-testing correction is performed independently for each:

        dataset × model × shot × cap

    Parameters
    ----------
    results_dir:
        Root containing Track C JSONL result files.
    metric:
        Per-document metric to compare, typically ``f1``.
    out_path:
        Destination CSV path.
    alpha:
        Significance threshold applied to adjusted p-values.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(
            f"alpha must be between 0 and 1, got {alpha}."
        )

    results = load_all_results(
        results_dir
    )

    if results.empty:
        empty = pd.DataFrame()
        output_path = Path(out_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        empty.to_csv(
            output_path,
            index=False,
        )
        return empty

    required_columns = {
        *FAMILY_COLUMNS,
        "technique",
        "doc_id",
        metric,
    }

    missing = required_columns - set(
        results.columns
    )

    if missing:
        raise ValueError(
            "Significance analysis is missing required columns: "
            f"{sorted(missing)}"
        )

    rows = []

    for family_keys, family_group in results.groupby(
        FAMILY_COLUMNS,
        dropna=False,
    ):
        family_values = (
            family_keys
            if isinstance(family_keys, tuple)
            else (family_keys,)
        )

        family_metadata = dict(
            zip(
                FAMILY_COLUMNS,
                family_values,
            )
        )

        techniques = sorted(
            family_group["technique"]
            .dropna()
            .unique()
        )

        for technique_a, technique_b in combinations(
            techniques,
            2,
        ):
            scores_a = _prepare_strategy_scores(
                family_group,
                technique_a,
                metric,
            )

            scores_b = _prepare_strategy_scores(
                family_group,
                technique_b,
                metric,
            )

            paired = scores_a.merge(
                scores_b,
                on="doc_id",
                how="inner",
                suffixes=("_a", "_b"),
            ).dropna(
                subset=[
                    f"{metric}_a",
                    f"{metric}_b",
                ]
            )

            if len(paired) < 2:
                continue

            x = paired[
                f"{metric}_a"
            ]

            y = paired[
                f"{metric}_b"
            ]

            statistic, p_value = _paired_wilcoxon(
                x,
                y,
            )

            mean_a = float(x.mean())
            mean_b = float(y.mean())
            mean_difference = mean_a - mean_b

            rows.append(
                {
                    **family_metadata,
                    "technique_a": technique_a,
                    "technique_b": technique_b,
                    "metric": metric,
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_difference": mean_difference,
                    "better_technique": (
                        technique_a
                        if mean_difference > 0
                        else technique_b
                        if mean_difference < 0
                        else "tie"
                    ),
                    "n_pairs": int(
                        len(paired)
                    ),
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )

    output = pd.DataFrame(rows)

    if not output.empty:
        output["p_adjusted_bh"] = np.nan

        # Apply Benjamini-Hochberg independently inside each statistical family.
        for _, family_indices in output.groupby(
            FAMILY_COLUMNS,
            dropna=False,
        ).groups.items():
            family_indices = list(
                family_indices
            )

            adjusted = benjamini_hochberg(
                output.loc[
                    family_indices,
                    "p_value",
                ].tolist()
            )

            output.loc[
                family_indices,
                "p_adjusted_bh",
            ] = adjusted

        output["significant"] = (
            output["p_adjusted_bh"] < alpha
        )

        output["alpha"] = alpha

        output["correction_method"] = (
            "Benjamini-Hochberg"
        )

        output["correction_family"] = (
            "dataset × model × shot × cap"
        )

        output = output.sort_values(
            FAMILY_COLUMNS
            + [
                "p_adjusted_bh",
                "technique_a",
                "technique_b",
            ],
            na_position="last",
        ).reset_index(
            drop=True
        )

    output_path = Path(out_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    return output


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run paired prompting-strategy significance tests with "
            "family-wise Benjamini-Hochberg correction."
        )
    )

    parser.add_argument(
        "--results",
        default="results",
    )

    parser.add_argument(
        "--metric",
        default="f1",
    )

    parser.add_argument(
        "--out",
        default="tables/significance_f1.csv",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    table = paired_strategy_tests(
        results_dir=args.results,
        metric=args.metric,
        out_path=args.out,
        alpha=args.alpha,
    )

    print(
        f"[written] {args.out} "
        f"({len(table)} comparisons)"
    )


if __name__ == "__main__":
    main()

