"""Aggregate latency metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_runtime_table(
    results_dir: str = "results",
    out_path: str = "tables/runtime.csv",
):
    rows = []

    for path in Path(results_dir).glob("*/*/*.meta.json"):
        dataset = path.parent.parent.name
        model = path.parent.name

        meta = json.loads(path.read_text(encoding="utf-8"))

        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "variant": meta.get("variant"),
                "mean_latency_s": meta.get("mean_latency_s"),
                "n_docs": meta.get("n_docs"),
                "cap_k": meta.get("cap_k"),
            }
        )

    df = pd.DataFrame(rows)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df