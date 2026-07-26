"""Shared utilities for loading Track C result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def parse_variant_slug(slug: str) -> dict:
    parts = slug.split("__")

    return {
        "technique": parts[0] if len(parts) > 0 else slug,
        "shot": parts[1] if len(parts) > 1 else "",
        "cap": parts[2] if len(parts) > 2 else "",
    }


def iter_result_files(results_dir: str = "results") -> Iterable[Path]:
    root = Path(results_dir)
    yield from sorted(root.glob("*/*/*.jsonl"))


def load_all_results(results_dir: str = "results") -> pd.DataFrame:
    frames = []

    for path in iter_result_files(results_dir):
        dataset = path.parent.parent.name
        model = path.parent.name
        variant = parse_variant_slug(path.stem)

        records = read_jsonl(path)
        if not records:
            continue

        df = pd.DataFrame(records)
        df["dataset"] = dataset
        df["model"] = model
        df["variant_slug"] = path.stem
        df["technique"] = variant["technique"]
        df["shot"] = variant["shot"]
        df["cap"] = variant["cap"]

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)