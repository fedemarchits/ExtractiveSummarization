"""Generate all Track C tables and figures from saved predictions.

    python -m analysis.run_all

    # BillSum run with the BillSum-oriented prompts (separate results tree)
    python -m analysis.run_all \
        --results results_updated_prompts \
        --tables tables_updated_prompts \
        --figures figures_updated_prompts \
        --dataset billsum
"""

from __future__ import annotations

import argparse

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

    parser = argparse.ArgumentParser(
        description="Generate all Track C tables and figures."
    )
    parser.add_argument(
        "--results",
        default="results",
        help="Results tree to analyse (results/<dataset>/<model>/...).",
    )
    parser.add_argument(
        "--tables",
        default="tables",
        help="Output directory for tables.",
    )
    parser.add_argument(
        "--figures",
        default="figures",
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Build the single-dataset tables (10, 11, 12) on this dataset. "
            "Default: Table 10 and 12 on xsum, Table 11 on cnndm."
        ),
    )
    parser.add_argument(
        "--model",
        default="qwen35_4b",
        help="Model used for the single-model tables (10, 11, 12).",
    )
    args = parser.parse_args()

    results = args.results
    tables = args.tables
    figures = args.figures

    print("[analysis] Tables 10 and 11")
    build_ablation_tables(
        results_dir=results,
        output_dir=tables,
        dataset=args.dataset,
        model=args.model,
    )

    print("[analysis] Table 12 and metric correlation")

    try:
        strategy_table, spearman_table, table12 = (
            run_correlation_analysis(
                results_dir=results,
                strategy_out=f"{tables}/metric_correlation.csv",
                spearman_out=f"{tables}/spearman_correlation.csv",
                table12_out=f"{tables}/table12_rank_invariance.csv",
                table12_dataset=args.dataset or "xsum",
                table12_model=args.model,
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
    build_bootstrap_table(
        results_dir=results,
        out_path=f"{tables}/bootstrap_confidence_intervals.csv",
    )

    print("[analysis] significance")
    paired_strategy_tests(
        results_dir=results,
        metric="f1",
        out_path=f"{tables}/significance_f1.csv",
    )

    print("[analysis] runtime")
    build_runtime_table(
        results_dir=results,
        out_path=f"{tables}/runtime.csv",
    )

    print("[analysis] position bias")
    positions = build_position_table(
        results_dir=results,
        out_path=f"{tables}/position_bias.csv",
    )

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
                    f"{figures}/position_bias_"
                    f"{dataset}_{model}.pdf"
                ),
            )

    print("[analysis] complete")


if __name__ == "__main__":
    main()