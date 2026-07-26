"""ROUGE-2 and MoverScore correlation analysis for Track C.

Input files:

    results/<dataset>/<model>/<variant_slug>.jsonl

Each result row must contain:

    doc_id
    pred_text
    reference_summary
    rouge2_model

The module:

1. computes or loads cached per-document MoverScore values;
2. aggregates mean ROUGE-2 and MoverScore per prompting variant;
3. ranks variants within comparable experimental conditions;
4. computes Spearman correlation within those conditions;
5. generates Table 12 for:
       vanilla
       explanation_based
       salience_inference
6. saves all generated tables as CSV files.

MoverScore remains optional. The ordinary Track C benchmark does not depend on
this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from scipy.stats import spearmanr

from .io import (
    iter_result_files,
    parse_variant_slug,
    read_jsonl,
)


try:
    from moverscore import (
        get_idf_dict,
        word_mover_score,
    )

    MOVERSCORE_AVAILABLE = True
    MOVERSCORE_IMPORT_ERROR: Optional[Exception] = None

except Exception as exc:
    MOVERSCORE_AVAILABLE = False
    MOVERSCORE_IMPORT_ERROR = exc
    get_idf_dict = None
    word_mover_score = None


DEFAULT_CACHE_DIR = "results_with_moverscore"

COMPARISON_COLUMNS = [
    "dataset",
    "model",
    "shot",
    "cap",
]

STRATEGY_COLUMNS = [
    "dataset",
    "model",
    "technique",
    "shot",
    "cap",
    "variant_slug",
]

TABLE12_TECHNIQUES = [
    "vanilla",
    "explanation_based",
    "salience_inference",
]

TABLE12_NAMES = {
    "vanilla": "Vanilla Selection",
    "explanation_based": "Explanation-Based (EBS)",
    "salience_inference": "Salience Inference (SIP)",
}


def _safe_text(value) -> str:
    """Convert a potentially missing value into clean text."""
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source result file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _write_jsonl(
    records: Sequence[Dict],
    path: Path,
) -> None:
    """Write records atomically as JSONL."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    partial_path = path.with_suffix(
        path.suffix + ".partial"
    )

    with partial_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    partial_path.replace(path)


def _cache_paths(
    result_path: Path,
    results_root: Path,
    cache_root: Path,
) -> Tuple[Path, Path]:
    """Return the cached JSONL and metadata paths."""
    relative_path = result_path.relative_to(
        results_root
    )

    cached_jsonl = cache_root / relative_path

    cached_meta = cached_jsonl.with_suffix(
        ".moverscore.meta.json"
    )

    return cached_jsonl, cached_meta


def _cache_is_valid(
    result_path: Path,
    cached_jsonl: Path,
    cached_meta: Path,
) -> bool:
    """Return whether cached MoverScore data matches the source JSONL."""
    if not cached_jsonl.exists():
        return False

    if not cached_meta.exists():
        return False

    try:
        metadata = json.loads(
            cached_meta.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return False

    source_stat = result_path.stat()

    if metadata.get("source_size_bytes") != source_stat.st_size:
        return False

    if metadata.get("source_mtime_ns") != source_stat.st_mtime_ns:
        return False

    if metadata.get("source_sha256") != _file_sha256(result_path):
        return False

    try:
        cached_records = read_jsonl(
            cached_jsonl
        )
    except Exception:
        return False

    if not cached_records:
        return False

    # Missing/empty hypotheses legitimately receive a None MoverScore.
    # The field must exist, but its value does not need to be numeric.
    return all(
        "moverscore" in record
        for record in cached_records
    )


def _write_cache_metadata(
    result_path: Path,
    cached_meta: Path,
    device_used: str,
    n_records: int,
) -> None:
    """Write metadata used to validate one cache file."""
    source_stat = result_path.stat()

    metadata = {
        "source_path": str(
            result_path.resolve()
        ),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "source_sha256": _file_sha256(
            result_path
        ),
        "device_used": device_used,
        "n_records": int(n_records),
    }

    cached_meta.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cached_meta.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def compute_moverscore(
    references: List[str],
    hypotheses: List[str],
    device: str = "cuda:0",
    batch_size: int = 16,
) -> List[float]:
    """Compute one MoverScore value per reference-hypothesis pair."""
    if not MOVERSCORE_AVAILABLE:
        raise RuntimeError(
            "MoverScore could not be imported. "
            f"Original import error: {MOVERSCORE_IMPORT_ERROR}"
        )

    if len(references) != len(hypotheses):
        raise ValueError(
            "references and hypotheses must have the same length."
        )

    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}."
        )

    if not references:
        return []

    idf_reference = get_idf_dict(
        references
    )

    idf_hypothesis = get_idf_dict(
        hypotheses
    )

    scores = word_mover_score(
        refs=references,
        hyps=hypotheses,
        idf_ref=idf_reference,
        idf_hyp=idf_hypothesis,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
        batch_size=batch_size,
        device=device,
    )

    return [
        float(score)
        for score in scores
    ]


def _compute_with_fallback(
    references: List[str],
    hypotheses: List[str],
    device: str,
    batch_size: int,
) -> Tuple[List[float], str]:
    """Compute MoverScore and retry on CPU after a GPU failure."""
    try:
        scores = compute_moverscore(
            references=references,
            hypotheses=hypotheses,
            device=device,
            batch_size=batch_size,
        )

        return scores, device

    except Exception as preferred_device_error:
        if device == "cpu":
            raise

        print(
            f"[warning] MoverScore failed on {device}: "
            f"{preferred_device_error}"
        )
        print(
            "[warning] Retrying MoverScore on CPU."
        )

        cpu_batch_size = max(
            1,
            min(batch_size, 8),
        )

        scores = compute_moverscore(
            references=references,
            hypotheses=hypotheses,
            device="cpu",
            batch_size=cpu_batch_size,
        )

        return scores, "cpu"


def compute_file_scores(
    result_path: Path,
    results_root: Path,
    cache_root: Path,
    device: str = "cuda:0",
    batch_size: int = 16,
    overwrite_cache: bool = False,
) -> pd.DataFrame:
    """Load or calculate per-document MoverScore for one result file.

    The original result file is not modified. Enriched rows are saved under the
    parallel MoverScore cache directory.
    """
    result_path = result_path.resolve()
    results_root = results_root.resolve()
    cache_root = cache_root.resolve()

    cached_jsonl, cached_meta = _cache_paths(
        result_path=result_path,
        results_root=results_root,
        cache_root=cache_root,
    )

    if (
        not overwrite_cache
        and _cache_is_valid(
            result_path=result_path,
            cached_jsonl=cached_jsonl,
            cached_meta=cached_meta,
        )
    ):
        print(
            f"[cache] {cached_jsonl}"
        )

        return pd.DataFrame(
            read_jsonl(cached_jsonl)
        )

    records = read_jsonl(
        result_path
    )

    if not records:
        return pd.DataFrame()

    required_fields = {
        "doc_id",
        "pred_text",
        "reference_summary",
        "rouge2_model",
    }

    available_fields = set().union(
        *(
            set(record)
            for record in records
            if isinstance(record, dict)
        )
    )

    missing_fields = (
        required_fields
        - available_fields
    )

    if missing_fields:
        raise ValueError(
            f"{result_path} is missing fields required for "
            f"MoverScore analysis: {sorted(missing_fields)}"
        )

    references: List[str] = []
    hypotheses: List[str] = []
    valid_record_indices: List[int] = []

    for record_index, record in enumerate(
        records
    ):
        hypothesis = _safe_text(
            record.get("pred_text")
        )

        reference = _safe_text(
            record.get("reference_summary")
        )

        if hypothesis and reference:
            valid_record_indices.append(
                record_index
            )
            hypotheses.append(
                hypothesis
            )
            references.append(
                reference
            )

    moverscores: List[Optional[float]] = [
        None
    ] * len(records)

    device_used = device

    if references:
        calculated_scores, device_used = (
            _compute_with_fallback(
                references=references,
                hypotheses=hypotheses,
                device=device,
                batch_size=batch_size,
            )
        )

        if len(calculated_scores) != len(
            valid_record_indices
        ):
            raise RuntimeError(
                "MoverScore returned a different number of scores "
                "from the number of valid document pairs."
            )

        for record_index, score in zip(
            valid_record_indices,
            calculated_scores,
        ):
            moverscores[record_index] = float(
                score
            )

    for record, score in zip(
        records,
        moverscores,
    ):
        record["moverscore"] = score
        record["moverscore_device"] = (
            device_used
        )

    _write_jsonl(
        records=records,
        path=cached_jsonl,
    )

    _write_cache_metadata(
        result_path=result_path,
        cached_meta=cached_meta,
        device_used=device_used,
        n_records=len(records),
    )

    print(
        f"[cached] {cached_jsonl}"
    )

    return pd.DataFrame(records)


def build_strategy_table(
    results_dir: str = "results",
    cache_dir: str = DEFAULT_CACHE_DIR,
    device: str = "cuda:0",
    batch_size: int = 16,
    overwrite_cache: bool = False,
) -> pd.DataFrame:
    """Build one aggregate row per complete prompting variant."""
    results_root = Path(
        results_dir
    ).resolve()

    cache_root = Path(
        cache_dir
    ).resolve()

    frames: List[pd.DataFrame] = []

    for result_file in iter_result_files(
        results_dir
    ):
        result_path = Path(
            result_file
        ).resolve()

        dataset = result_path.parent.parent.name
        model = result_path.parent.name

        variant = parse_variant_slug(
            result_path.stem
        )

        print(
            f"[moverscore] "
            f"{dataset}/{model}/{result_path.stem}"
        )

        dataframe = compute_file_scores(
            result_path=result_path,
            results_root=results_root,
            cache_root=cache_root,
            device=device,
            batch_size=batch_size,
            overwrite_cache=overwrite_cache,
        )

        if dataframe.empty:
            continue

        required_columns = {
            "doc_id",
            "rouge2_model",
            "moverscore",
        }

        missing_columns = (
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{result_path} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        dataframe["dataset"] = dataset
        dataframe["model"] = model
        dataframe["variant_slug"] = (
            result_path.stem
        )
        dataframe["technique"] = variant[
            "technique"
        ]
        dataframe["shot"] = variant["shot"]
        dataframe["cap"] = variant["cap"]

        frames.append(
            dataframe
        )

    if not frames:
        return pd.DataFrame()

    all_documents = pd.concat(
        frames,
        ignore_index=True,
    )

    all_documents["rouge2_model"] = (
        pd.to_numeric(
            all_documents["rouge2_model"],
            errors="coerce",
        )
    )

    all_documents["moverscore"] = (
        pd.to_numeric(
            all_documents["moverscore"],
            errors="coerce",
        )
    )

    strategy_table = (
        all_documents.groupby(
            STRATEGY_COLUMNS,
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_rouge2=(
                "rouge2_model",
                "mean",
            ),
            mean_moverscore=(
                "moverscore",
                "mean",
            ),
            n_docs=(
                "doc_id",
                "nunique",
            ),
            n_valid_rouge2=(
                "rouge2_model",
                "count",
            ),
            n_valid_moverscore=(
                "moverscore",
                "count",
            ),
        )
    )

    # Rank only within equivalent experimental conditions.
    strategy_table["rouge_rank"] = (
        strategy_table.groupby(
            COMPARISON_COLUMNS,
            dropna=False,
        )["mean_rouge2"]
        .rank(
            method="average",
            ascending=False,
        )
    )

    strategy_table["moverscore_rank"] = (
        strategy_table.groupby(
            COMPARISON_COLUMNS,
            dropna=False,
        )["mean_moverscore"]
        .rank(
            method="average",
            ascending=False,
        )
    )

    strategy_table["rank_difference"] = (
        strategy_table["rouge_rank"]
        - strategy_table["moverscore_rank"]
    )

    return strategy_table.sort_values(
        COMPARISON_COLUMNS
        + [
            "rouge_rank",
            "technique",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )


def build_spearman_table(
    strategy_table: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Spearman correlation across comparable strategies.

    Each correlation family is:

        dataset × model × shot × cap
    """
    rows: List[Dict] = []

    if strategy_table.empty:
        return pd.DataFrame()

    for keys, group in strategy_table.groupby(
        COMPARISON_COLUMNS,
        dropna=False,
    ):
        family_values = (
            keys
            if isinstance(keys, tuple)
            else (keys,)
        )

        family = dict(
            zip(
                COMPARISON_COLUMNS,
                family_values,
            )
        )

        valid = group.dropna(
            subset=[
                "mean_rouge2",
                "mean_moverscore",
            ]
        )

        if (
            len(valid) < 2
            or valid["mean_rouge2"].nunique() < 2
            or valid["mean_moverscore"].nunique() < 2
        ):
            rho = float("nan")
            p_value = float("nan")

        else:
            correlation = spearmanr(
                valid["mean_rouge2"],
                valid["mean_moverscore"],
            )

            rho = float(
                correlation.statistic
            )

            p_value = float(
                correlation.pvalue
            )

        rows.append(
            {
                **family,
                "n_strategies": int(
                    len(valid)
                ),
                "spearman_rho": rho,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows).sort_values(
        COMPARISON_COLUMNS
    ).reset_index(
        drop=True
    )


def build_table12_rank_invariance(
    strategy_table: pd.DataFrame,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    shot: Optional[str] = None,
    cap: Optional[str] = None,
    out_path: str = "tables/table12_rank_invariance.csv",
) -> pd.DataFrame:
    """Build Table 12 for all matching experimental conditions.

    When dataset, model, shot, or cap is omitted, all available values for that
    dimension are retained.

    Each table family contains only:

        vanilla
        explanation_based
        salience_inference

    Rankings are recalculated among those three strategies only.
    """
    if strategy_table.empty:
        output = pd.DataFrame()
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

    subset = strategy_table[
        strategy_table["technique"].isin(
            TABLE12_TECHNIQUES
        )
    ].copy()

    if dataset is not None:
        subset = subset[
            subset["dataset"] == dataset
        ]

    if model is not None:
        subset = subset[
            subset["model"] == model
        ]

    if shot is not None:
        subset = subset[
            subset["shot"] == shot
        ]

    if cap is not None:
        subset = subset[
            subset["cap"] == cap
        ]

    if subset.empty:
        requested = {
            "dataset": dataset,
            "model": model,
            "shot": shot,
            "cap": cap,
        }

        raise RuntimeError(
            "No strategy rows matched the requested Table 12 "
            f"condition: {requested}"
        )

    completed_families: List[pd.DataFrame] = []
    incomplete_families: List[Dict] = []

    for keys, family in subset.groupby(
        COMPARISON_COLUMNS,
        dropna=False,
    ):
        family_values = (
            keys
            if isinstance(keys, tuple)
            else (keys,)
        )

        family_metadata = dict(
            zip(
                COMPARISON_COLUMNS,
                family_values,
            )
        )

        present = set(
            family["technique"]
        )

        missing = (
            set(TABLE12_TECHNIQUES)
            - present
        )

        if missing:
            incomplete_families.append(
                {
                    **family_metadata,
                    "missing": sorted(missing),
                }
            )
            continue

        # Protect against accidental duplicates.
        family = (
            family.sort_values(
                "variant_slug"
            )
            .drop_duplicates(
                subset=["technique"],
                keep="first",
            )
            .copy()
        )

        family["rouge_rank_table12"] = (
            family["mean_rouge2"]
            .rank(
                method="average",
                ascending=False,
            )
        )

        family["moverscore_rank_table12"] = (
            family["mean_moverscore"]
            .rank(
                method="average",
                ascending=False,
            )
        )

        family["prompting_strategy"] = (
            family["technique"]
            .map(TABLE12_NAMES)
        )

        completed_families.append(
            family
        )

    if not completed_families:
        raise RuntimeError(
            "Table 12 could not be generated because no experimental "
            "condition contained all three required techniques. "
            f"Incomplete families: {incomplete_families}"
        )

    if incomplete_families:
        print(
            "[warning] Some Table 12 conditions were skipped because "
            "they did not contain all three strategies:"
        )

        for family in incomplete_families:
            print(
                f"  {family}"
            )

    output = pd.concat(
        completed_families,
        ignore_index=True,
    )

    output = output[
        [
            "dataset",
            "model",
            "shot",
            "cap",
            "prompting_strategy",
            "technique",
            "mean_rouge2",
            "mean_moverscore",
            "rouge_rank_table12",
            "moverscore_rank_table12",
            "n_docs",
            "n_valid_moverscore",
        ]
    ].sort_values(
        COMPARISON_COLUMNS
        + [
            "rouge_rank_table12",
            "technique",
        ]
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


def run_correlation_analysis(
    results_dir: str = "results",
    cache_dir: str = DEFAULT_CACHE_DIR,
    strategy_out: str = "tables/metric_correlation.csv",
    spearman_out: str = "tables/spearman_correlation.csv",
    table12_out: str = "tables/table12_rank_invariance.csv",
    device: str = "cuda:0",
    batch_size: int = 16,
    overwrite_cache: bool = False,
    table12_dataset: Optional[str] = None,
    table12_model: Optional[str] = None,
    table12_shot: Optional[str] = None,
    table12_cap: Optional[str] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the complete Track C metric-correlation analysis."""
    if not MOVERSCORE_AVAILABLE:
        raise RuntimeError(
            "MoverScore is unavailable. "
            f"Import error: {MOVERSCORE_IMPORT_ERROR}"
        )

    strategy_table = build_strategy_table(
        results_dir=results_dir,
        cache_dir=cache_dir,
        device=device,
        batch_size=batch_size,
        overwrite_cache=overwrite_cache,
    )

    spearman_table = build_spearman_table(
        strategy_table
    )

    table12 = build_table12_rank_invariance(
        strategy_table=strategy_table,
        dataset=table12_dataset,
        model=table12_model,
        shot=table12_shot,
        cap=table12_cap,
        out_path=table12_out,
    )

    strategy_path = Path(
        strategy_out
    )
    strategy_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    strategy_table.to_csv(
        strategy_path,
        index=False,
    )

    spearman_path = Path(
        spearman_out
    )
    spearman_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    spearman_table.to_csv(
        spearman_path,
        index=False,
    )

    print(
        f"[written] {strategy_path} "
        f"({len(strategy_table)} rows)"
    )

    print(
        f"[written] {spearman_path} "
        f"({len(spearman_table)} rows)"
    )

    print(
        f"[written] {table12_out} "
        f"({len(table12)} rows)"
    )

    return (
        strategy_table,
        spearman_table,
        table12,
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute cached MoverScore values, strategy-level metric "
            "correlations, and Table 12 rank invariance."
        )
    )

    parser.add_argument(
        "--results",
        default="results",
        help="Root directory containing Track C JSONL result files.",
    )

    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Parallel directory for per-document MoverScore caches.",
    )

    parser.add_argument(
        "--strategy-out",
        default="tables/metric_correlation.csv",
    )

    parser.add_argument(
        "--spearman-out",
        default="tables/spearman_correlation.csv",
    )

    parser.add_argument(
        "--table12-out",
        default="tables/table12_rank_invariance.csv",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="Recompute MoverScore even when a valid cache exists.",
    )

    parser.add_argument(
        "--table12-dataset",
        choices=["xsum", "cnndm"],
        default=None,
        help=(
            "Optional dataset filter for Table 12. "
            "Omit to include both datasets."
        ),
    )

    parser.add_argument(
        "--table12-model",
        default=None,
        help=(
            "Optional model-alias filter for Table 12. "
            "Omit to include every available model."
        ),
    )

    parser.add_argument(
        "--table12-shot",
        choices=["zero_shot", "one_shot"],
        default=None,
        help=(
            "Optional shot-setting filter for Table 12. "
            "Omit to include both settings."
        ),
    )

    parser.add_argument(
        "--table12-cap",
        choices=["capped", "uncapped"],
        default=None,
        help=(
            "Optional cap-setting filter for Table 12. "
            "Omit to include both settings."
        ),
    )

    args = parser.parse_args()

    run_correlation_analysis(
        results_dir=args.results,
        cache_dir=args.cache_dir,
        strategy_out=args.strategy_out,
        spearman_out=args.spearman_out,
        table12_out=args.table12_out,
        device=args.device,
        batch_size=args.batch_size,
        overwrite_cache=args.overwrite_cache,
        table12_dataset=args.table12_dataset,
        table12_model=args.table12_model,
        table12_shot=args.table12_shot,
        table12_cap=args.table12_cap,
    )


if __name__ == "__main__":
    main()





# """ROUGE-2 and MoverScore correlation analysis for Track C.

# Reads saved prediction files from:

#     results/<dataset>/<model>/<variant_slug>.jsonl

# Each JSONL row must contain:
# - doc_id
# - pred_text
# - reference_summary
# - rouge2_model

# The script:
# 1. computes one MoverScore value per document;
# 2. averages ROUGE-2 and MoverScore per prompting strategy;
# 3. ranks strategies under both metrics;
# 4. computes Spearman correlation across strategies;
# 5. saves two CSV files.

# This module is optional. The rest of Track C does not depend on MoverScore.
# """

# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import pandas as pd
# from scipy.stats import spearmanr

# from .io import iter_result_files, parse_variant_slug, read_jsonl


# try:
#     from moverscore import get_idf_dict, word_mover_score

#     MOVERSCORE_AVAILABLE = True
#     MOVERSCORE_IMPORT_ERROR: Optional[Exception] = None
# except Exception as exc:
#     MOVERSCORE_AVAILABLE = False
#     MOVERSCORE_IMPORT_ERROR = exc
#     get_idf_dict = None
#     word_mover_score = None

# TABLE12_TECHNIQUES = [
#     "vanilla",
#     "explanation_based",
#     "salience_inference",
# ]

# TABLE12_NAMES = {
#     "vanilla": "Vanilla Selection",
#     "explanation_based": "Explanation-Based (EBS)",
#     "salience_inference": "Salience Inference (SIP)",
# }

# def build_table12_rank_invariance(
#     strategy_table: pd.DataFrame,
#     dataset: str,
#     model: str,
#     shot: str,
#     cap: str,
#     out_path: str = "tables/table12_rank_invariance.csv",
# ) -> pd.DataFrame:
#     """Build Table 12 for one comparable experimental condition."""
#     subset = strategy_table[
#         (strategy_table["dataset"] == dataset)
#         & (strategy_table["model"] == model)
#         & (strategy_table["shot"] == shot)
#         & (strategy_table["cap"] == cap)
#         & (
#             strategy_table["technique"].isin(
#                 TABLE12_TECHNIQUES
#             )
#         )
#     ].copy()

#     if set(subset["technique"]) != set(
#         TABLE12_TECHNIQUES
#     ):
#         missing = set(TABLE12_TECHNIQUES) - set(
#             subset["technique"]
#         )

#         raise RuntimeError(
#             f"Table 12 is missing strategies: {sorted(missing)}"
#         )

#     subset["rouge_rank"] = subset[
#         "mean_rouge2"
#     ].rank(
#         method="average",
#         ascending=False,
#     )

#     subset["moverscore_rank"] = subset[
#         "mean_moverscore"
#     ].rank(
#         method="average",
#         ascending=False,
#     )

#     subset["prompting_strategy"] = subset[
#         "technique"
#     ].map(TABLE12_NAMES)

#     output = subset[
#         [
#             "prompting_strategy",
#             "technique",
#             "mean_rouge2",
#             "mean_moverscore",
#             "rouge_rank",
#             "moverscore_rank",
#         ]
#     ].sort_values(
#         "rouge_rank"
#     )

#     path = Path(out_path)
#     path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     output.to_csv(
#         path,
#         index=False,
#     )

#     return output


# def compute_moverscore(
#     references: List[str],
#     hypotheses: List[str],
#     device: str = "cuda:0",
#     batch_size: int = 16,
# ) -> List[float]:
#     """Compute one MoverScore value per reference-hypothesis pair."""
#     if not MOVERSCORE_AVAILABLE:
#         raise RuntimeError(
#             "MoverScore could not be imported. "
#             f"Original error: {MOVERSCORE_IMPORT_ERROR}"
#         )

#     if len(references) != len(hypotheses):
#         raise ValueError(
#             "references and hypotheses must have the same length."
#         )

#     if not references:
#         return []

#     idf_ref = get_idf_dict(references)
#     idf_hyp = get_idf_dict(hypotheses)

#     scores = word_mover_score(
#         refs=references,
#         hyps=hypotheses,
#         idf_ref=idf_ref,
#         idf_hyp=idf_hyp,
#         stop_words=[],
#         n_gram=1,
#         remove_subwords=True,
#         batch_size=batch_size,
#         device=device,
#     )

#     return [float(score) for score in scores]


# def compute_file_scores(
#     result_path: Path,
#     device: str = "cuda:0",
#     batch_size: int = 16,
# ) -> pd.DataFrame:
#     """Load one JSONL result file and add per-document MoverScore."""
#     records = read_jsonl(result_path)

#     if not records:
#         return pd.DataFrame()

#     references: List[str] = []
#     hypotheses: List[str] = []
#     valid_record_indices: List[int] = []

#     for i, record in enumerate(records):
#         pred_text = str(record.get("pred_text", "") or "").strip()
#         reference = str(
#             record.get("reference_summary", "") or ""
#         ).strip()

#         if pred_text and reference:
#             valid_record_indices.append(i)
#             hypotheses.append(pred_text)
#             references.append(reference)

#     moverscores = [None] * len(records)

#     if references:
#         try:
#             computed = compute_moverscore(
#                 references=references,
#                 hypotheses=hypotheses,
#                 device=device,
#                 batch_size=batch_size,
#             )
#         except Exception as gpu_error:
#             if device == "cpu":
#                 raise

#             print(
#                 f"[warning] MoverScore failed on {device}: "
#                 f"{gpu_error}"
#             )
#             print("[warning] Retrying on CPU.")

#             computed = compute_moverscore(
#                 references=references,
#                 hypotheses=hypotheses,
#                 device="cpu",
#                 batch_size=min(batch_size, 8),
#             )

#         for record_idx, score in zip(
#             valid_record_indices,
#             computed,
#         ):
#             moverscores[record_idx] = score

#     for record, score in zip(records, moverscores):
#         record["moverscore"] = score

#     return pd.DataFrame(records)


# def build_strategy_table(
#     results_dir: str = "results",
#     device: str = "cuda:0",
#     batch_size: int = 16,
# ) -> pd.DataFrame:
#     """Build one aggregated row per dataset/model/strategy."""
#     frames: List[pd.DataFrame] = []

#     for result_path in iter_result_files(results_dir):
#         dataset = result_path.parent.parent.name
#         model = result_path.parent.name
#         variant = parse_variant_slug(result_path.stem)

#         print(
#             f"[moverscore] {dataset}/{model}/{result_path.stem}"
#         )

#         df = compute_file_scores(
#             result_path=result_path,
#             device=device,
#             batch_size=batch_size,
#         )

#         if df.empty:
#             continue

#         required = {
#             "doc_id",
#             "rouge2_model",
#             "moverscore",
#         }

#         missing = required - set(df.columns)

#         if missing:
#             raise ValueError(
#                 f"{result_path} is missing columns: "
#                 f"{sorted(missing)}"
#             )

#         df["dataset"] = dataset
#         df["model"] = model
#         df["variant_slug"] = result_path.stem
#         df["technique"] = variant["technique"]
#         df["shot"] = variant["shot"]
#         df["cap"] = variant["cap"]

#         frames.append(df)

#     if not frames:
#         return pd.DataFrame()

#     all_docs = pd.concat(frames, ignore_index=True)

#     strategy_table = (
#         all_docs.groupby(
#             [
#                 "dataset",
#                 "model",
#                 "technique",
#                 "shot",
#                 "cap",
#                 "variant_slug",
#             ],
#             as_index=False,
#         )
#         .agg(
#             mean_rouge2=("rouge2_model", "mean"),
#             mean_moverscore=("moverscore", "mean"),
#             n_docs=("doc_id", "nunique"),
#             n_valid_moverscore=("moverscore", "count"),
#         )
#     )

#     strategy_table["rouge_rank"] = (
#         strategy_table.groupby(
#             ["dataset", "model"]
#         )["mean_rouge2"]
#         .rank(
#             method="average",
#             ascending=False,
#         )
#     )

#     strategy_table["moverscore_rank"] = (
#         strategy_table.groupby(
#             ["dataset", "model"]
#         )["mean_moverscore"]
#         .rank(
#             method="average",
#             ascending=False,
#         )
#     )

#     strategy_table["rank_difference"] = (
#         strategy_table["rouge_rank"]
#         - strategy_table["moverscore_rank"]
#     )

#     return strategy_table.sort_values(
#         [
#             "dataset",
#             "model",
#             "rouge_rank",
#         ]
#     ).reset_index(drop=True)


# def build_spearman_table(
#     strategy_table: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Compute Spearman correlation across strategies."""
#     rows: List[Dict] = []

#     if strategy_table.empty:
#         return pd.DataFrame()

#     for (dataset, model), group in strategy_table.groupby(
#         ["dataset", "model"]
#     ):
#         valid = group.dropna(
#             subset=[
#                 "mean_rouge2",
#                 "mean_moverscore",
#             ]
#         )

#         if (
#             len(valid) < 2
#             or valid["mean_rouge2"].nunique() < 2
#             or valid["mean_moverscore"].nunique() < 2
#         ):
#             rho = float("nan")
#             p_value = float("nan")
#         else:
#             result = spearmanr(
#                 valid["mean_rouge2"],
#                 valid["mean_moverscore"],
#             )

#             rho = float(result.statistic)
#             p_value = float(result.pvalue)

#         rows.append(
#             {
#                 "dataset": dataset,
#                 "model": model,
#                 "n_strategies": len(valid),
#                 "spearman_rho": rho,
#                 "p_value": p_value,
#             }
#         )

#     return pd.DataFrame(rows)


# def run_correlation_analysis(
#     results_dir: str = "results",
#     strategy_out: str = "tables/table13_metric_correlation.csv",
#     spearman_out: str = "tables/table13_spearman.csv",
#     device: str = "cuda:0",
#     batch_size: int = 16,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Run the full Track C correlation analysis."""
#     if not MOVERSCORE_AVAILABLE:
#         raise RuntimeError(
#             "MoverScore is unavailable. "
#             f"Import error: {MOVERSCORE_IMPORT_ERROR}"
#         )

#     strategy_table = build_strategy_table(
#         results_dir=results_dir,
#         device=device,
#         batch_size=batch_size,
#     )

#     spearman_table = build_spearman_table(
#         strategy_table
#     )

#     strategy_path = Path(strategy_out)
#     strategy_path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#     strategy_table.to_csv(
#         strategy_path,
#         index=False,
#     )

#     spearman_path = Path(spearman_out)
#     spearman_path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#     spearman_table.to_csv(
#         spearman_path,
#         index=False,
#     )

#     print(
#         f"[written] {strategy_path} "
#         f"({len(strategy_table)} rows)"
#     )
#     print(
#         f"[written] {spearman_path} "
#         f"({len(spearman_table)} rows)"
#     )

#     return strategy_table, spearman_table


# def main() -> None:
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--results",
#         default="results",
#     )
#     parser.add_argument(
#         "--strategy-out",
#         default="tables/table13_metric_correlation.csv",
#     )
#     parser.add_argument(
#         "--spearman-out",
#         default="tables/table13_spearman.csv",
#     )
#     parser.add_argument(
#         "--device",
#         default="cuda:0",
#     )
#     parser.add_argument(
#         "--batch-size",
#         type=int,
#         default=16,
#     )

#     args = parser.parse_args()

#     run_correlation_analysis(
#         results_dir=args.results,
#         strategy_out=args.strategy_out,
#         spearman_out=args.spearman_out,
#         device=args.device,
#         batch_size=args.batch_size,
#     )


# if __name__ == "__main__":
#     main()