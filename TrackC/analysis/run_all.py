"""Generate all Track C tables and figures from saved predictions."""

from __future__ import annotations

from .ablations import build_ablation_tables
from .bootstrap import build_bootstrap_table
from .correlation import run_correlation_analysis
from .position_bias import (
    build_position_table,
    plot_position_bias,
)
from .runtime import build_runtime_table
from .significance import paired_strategy_tests


def main() -> None:
    """Run all Track C post-processing analyses."""

    print("[analysis] Tables 10 and 11")
    build_ablation_tables()

    print("[analysis] Table 12 and metric correlation")

    try:
        strategy_table, spearman_table, table12 = (
            run_correlation_analysis(
                table12_dataset="xsum",
                table12_model="qwen35_4b",
                table12_shot="zero_shot",
                table12_cap="capped",
            )
        )

        print(
            "[analysis] correlation outputs: "
            f"{len(strategy_table)} strategy rows, "
            f"{len(spearman_table)} Spearman rows, "
            f"{len(table12)} Table 12 rows"
        )

    except RuntimeError as exc:
        # MoverScore is optional. Its failure should not prevent the remaining
        # Track C analyses from running.
        print(
            "[warning] Skipping MoverScore correlation and Table 12: "
            f"{exc}"
        )

    print("[analysis] bootstrap confidence intervals")
    build_bootstrap_table()

    print("[analysis] significance")
    paired_strategy_tests(
        metric="f1",
        out_path="tables/significance_f1.csv",
    )

    print("[analysis] runtime")
    build_runtime_table()

    print("[analysis] position bias")
    positions = build_position_table()

    if not positions.empty:
        # Only real model rows should determine which figures are generated.
        # The silver-reference curve is included inside each model figure by
        # plot_position_bias().
        model_rows = positions[
            positions["selection_source"] == "model"
        ]

        combinations = (
            model_rows[
                [
                    "dataset",
                    "model",
                ]
            ]
            .drop_duplicates()
            .itertuples(
                index=False,
                name=None,
            )
        )

        for dataset, model in combinations:
            plot_position_bias(
                positions,
                dataset=dataset,
                model=model,
                out_path=(
                    f"figures/position_bias_"
                    f"{dataset}_{model}.pdf"
                ),
            )

    print("[analysis] complete")


if __name__ == "__main__":
    main()