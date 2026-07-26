"""Bootstrap confidence intervals for Track C metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_all_results


def bootstrap_mean_ci(
    values,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(n_bootstrap, len(values)),
        replace=True,
    )

    means = samples.mean(axis=1)

    alpha = 1.0 - confidence
    lower = np.quantile(means, alpha / 2)
    upper = np.quantile(means, 1 - alpha / 2)

    return float(values.mean()), float(lower), float(upper)


def build_bootstrap_table(
    results_dir: str = "results",
    out_path: str = "tables/bootstrap_confidence_intervals.csv",
) -> pd.DataFrame:
    df = load_all_results(results_dir)
    rows = []

    group_cols = [
        "dataset",
        "model",
        "technique",
        "shot",
        "cap",
    ]

    for keys, grp in df.groupby(group_cols):
        base = dict(zip(group_cols, keys))

        for metric in ["precision", "recall", "f1"]:
            mean, low, high = bootstrap_mean_ci(grp[metric])

            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "n_docs": len(grp),
                }
            )

    out = pd.DataFrame(rows)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)

    return out