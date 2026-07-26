"""Aggregate Track C per-doc JSONL results into tidy summary tables.

Reads:
    results/<dataset>/<model>/<variant_slug>.jsonl

Emits one row per:
    dataset x model x variant

Track C has no ACLSum aspects. Each row in the JSONL corresponds to one
document from a silver XSum / CNN-DM dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


_METRICS = [
    "precision",
    "recall",
    "f1",
    "rouge1_model",
    "rouge2_model",
    "rougeL_model",
    "rouge1_oracle",
    "rouge2_oracle",
    "rougeL_oracle",
    "oracle_gap_rougeL",
]


def _read_jsonl(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _safe_slug_parts(slug: str):
    parts = slug.split("__")
    technique = parts[0] if len(parts) > 0 else slug
    shot = parts[1] if len(parts) > 1 else ""
    cap = parts[2] if len(parts) > 2 else ""
    return technique, shot, cap


def summarize(results_dir: str = "results") -> "pandas.DataFrame":  # noqa: F821
    import pandas as pd

    root = Path(results_dir)
    rows: List[Dict] = []

    for jsonl in sorted(root.glob("*/*/*.jsonl")):
        dataset = jsonl.parent.parent.name
        model = jsonl.parent.name
        slug = jsonl.stem

        technique, shot, cap = _safe_slug_parts(slug)

        recs = _read_jsonl(jsonl)
        if not recs:
            continue

        meta_path = jsonl.with_suffix(".meta.json")
        latency = None
        cap_k = None

        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            latency = meta.get("mean_latency_s")
            cap_k = meta.get("cap_k")

        df = pd.DataFrame(recs)

        row = {
            "dataset": dataset,
            "model": model,
            "technique": technique,
            "shot": shot,
            "cap": cap,
            "n": len(df),
            "mean_latency_s": latency,
            "cap_k": cap_k,
        }

        row.update(
            {
                k: float(df[k].mean())
                for k in _METRICS
                if k in df.columns
            }
        )

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    return (
        out.sort_values(["dataset", "model", "technique", "shot", "cap"])
        .reset_index(drop=True)
    )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/summary.csv")
    args = ap.parse_args()

    df = summarize(args.results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()