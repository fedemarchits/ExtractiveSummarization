"""Track C runner: model x prompt variant x document -> per-doc JSONL + metadata.

This runner reuses Track A prompting/model infrastructure, but evaluates on
Track C silver datasets generated from XSum / CNN-DailyMail using Track B's
best heuristic.

Differences from Track A:
- no ACLSum aspects
- no challenge/approach/outcome labels
- no union rows
- reference labels are silver_indices
- one JSONL row per document
- saves reconstructed prediction and silver summary text for later analyses
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from prompts.base import Cap, RenderCtx, Shot
from prompts.fewshot import select_exemplar
from prompts.registry import all_techniques

from . import data as D
from . import metrics as M
from .backends import GenConfig, get_backend
from .grid import resolve_variants
from .postprocess import cap_indices_in_order
from .wrappers import select_document

from prompts.rationale import configure_rationale_cache

SYSTEM = "You are an expert in extractive summarization."

# Track A prompt templates expect an aspect argument.
# Track C performs whole-document summarization, so this pseudo-aspect is used
# only to preserve compatibility with the Track A prompt interface.
TASK_ASPECT = "summary"


def _load_best_k(cfg: Dict) -> int:
    """Load the Track B selected sentence cap K.

    Priority:
    1. Explicit silver.best_k value in the experiment YAML.
    2. best_k stored in best_heuristic.json.

    Keeping the JSON as the main source of truth is recommended.
    """
    silver_cfg = cfg.get("silver", {})

    if "best_k" in silver_cfg:
        return int(silver_cfg["best_k"])

    heuristic_path = silver_cfg.get(
        "best_heuristic_path",
        "best_heuristic.json",
    )

    path = Path(heuristic_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Best-heuristic configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as fh:
        heuristic = json.load(fh)

    if "best_k" not in heuristic:
        raise KeyError(
            f"'best_k' is missing from heuristic configuration: {path}"
        )

    return int(heuristic["best_k"])

def _ctx_for(
    variant,
    cap_k: int,
    train=None,
    seed: int = 42,
) -> RenderCtx:
    """Build the rendering context for one Track C prompt variant.

    One-shot variants must have a valid exemplar from a separate validation
    silver dataset. They must never silently fall back to zero-shot.
    """
    k = cap_k if variant.cap is Cap.CAPPED else None

    exemplar = None

    if variant.shot is Shot.ONE:
        if train is None:
            raise RuntimeError(
                f"Variant {variant.slug!r} is configured as one-shot, "
                "but no validation silver dataset was loaded. "
                "Set dataset.train_silver_path in the experiment YAML "
                "and enable fewshot."
            )

        exemplar = select_exemplar(
            train,
            TASK_ASPECT,
            seed=seed,
        )

        if exemplar is None:
            raise RuntimeError(
                f"Variant {variant.slug!r} is configured as one-shot, "
                "but no valid exemplar could be selected."
            )

    return RenderCtx(
        shot=variant.shot,
        cap=variant.cap,
        k=k,
        exemplar=exemplar,
    )

# def _ctx_for(
#     variant,
#     cap_k: int,
#     train=None,
#     seed: int = 42,
# ) -> RenderCtx:
#     """Build a Track A-compatible rendering context for Track C.

#     Capped variants receive the Track B-selected K.

#     For one-shot variants, an exemplar is selected from the optional Track C
#     training/validation silver dataset. If no compatible exemplar dataset is
#     supplied, the prompt gracefully falls back to no exemplar.
#     """
#     k = cap_k if variant.cap is Cap.CAPPED else None

#     exemplar = None

#     if variant.shot is Shot.ONE and train is not None:
#         try:
#             exemplar = select_exemplar(
#                 train,
#                 TASK_ASPECT,
#                 seed=seed,
#             )
#         except Exception as exc:
#             print(
#                 "[warning] Could not select one-shot exemplar; "
#                 f"falling back to zero-shot context: {exc}"
#             )
#             exemplar = None

#     return RenderCtx(
#         shot=variant.shot,
#         cap=variant.cap,
#         k=k,
#         exemplar=exemplar,
#     )


def _parse_prediction(
    raw: str,
    n_sentences: int,
) -> List[int]:
    """Parse raw model output into valid unique 1-based sentence indices.

    Uses the shared postprocess pipeline (robust JSON extraction -> in-range
    unique sorted indices), the same path the wrapper code uses. No cap here;
    capped variants are capped later in run_variant.
    """
    from .postprocess import parse_selection

    return parse_selection(raw, n_sentences)


def _row(
    doc_id: str,
    doc_idx: int,
    dataset: str,
    n: int,
    pred: List[int],
    gold: List[int],
    pred_text: str,
    silver_text: str,
    reference_summary: str,
    metrics: Dict,
    rouge_model: Dict,
    oracle_idx: Optional[List[int]] = None,
    rouge_oracle: Optional[Dict] = None,
    raw: Optional[str] = None,
) -> Dict:
    """Construct one serializable per-document Track C result row."""
    rouge_oracle = rouge_oracle or {}
    oracle_idx = oracle_idx or []

    oracle_gap = (
        rouge_oracle.get("rougeL", 0.0)
        - rouge_model.get("rougeL", 0.0)
        if rouge_oracle
        else 0.0
    )

    return {
        "doc_id": doc_id,
        "doc_idx": doc_idx,
        "dataset": dataset,
        "num_sentences": n,

        "pred_length": len(pred),
        "silver_length": len(gold),

        # Sentence-index predictions and silver reference labels.
        "pred_indices": pred,
        "silver_indices": sorted(gold),

        # Reconstructed summaries saved for later analyses such as
        # MoverScore, qualitative inspection, and metric correlation.
        "pred_text": pred_text,
        "silver_text": silver_text,
        "reference_summary": reference_summary,

        # Exact index-match metrics.
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "f1": metrics.get("f1", 0.0),
        "tp": metrics.get("tp", 0),
        "fp": metrics.get("fp", 0),
        "fn": metrics.get("fn", 0),

        # ROUGE between model extractive output and abstractive reference.
        "rouge1_model": rouge_model.get("rouge1", 0.0),
        "rouge2_model": rouge_model.get("rouge2", 0.0),
        "rougeL_model": rouge_model.get("rougeL", 0.0),

        # Greedy oracle ROUGE against the same abstractive reference.
        "rouge1_oracle": rouge_oracle.get("rouge1", 0.0),
        "rouge2_oracle": rouge_oracle.get("rouge2", 0.0),
        "rougeL_oracle": rouge_oracle.get("rougeL", 0.0),
        "oracle_gap_rougeL": oracle_gap,
        "oracle_indices": oracle_idx,

        # Raw model output, retained for parsing-error analysis.
        "raw_response": raw,
    }


def run_variant(
    variant,
    backend,
    test,
    cap_k: int,
    seed: int,
    oracle_cache: Dict,
    out_path: Path,
    dataset_name: str,
    train=None,
    sc=None,
    dyn=None,
) -> None:
    """Run one prompt variant over one Track C silver dataset."""
    techniques = all_techniques()

    if variant.technique not in techniques:
        raise KeyError(
            f"Unknown prompting technique: {variant.technique}. "
            f"Available techniques: {sorted(techniques)}"
        )

    technique = techniques[variant.technique]

    system_prompt = (
    SYSTEM
    if technique.system_override is None
    else technique.system_override
)

    ctx = _ctx_for(
        variant=variant,
        cap_k=cap_k,
        train=train,
        seed=seed,
    )

    partial_path = out_path.with_suffix(".jsonl.partial")
    latencies: List[float] = []

    # vLLM-style backends benefit from receiving every prompt at once.
    # Wrapper methods such as self-consistency and dynamic capping require
    # per-document processing and therefore disable this fast path.
    fast_batch = (
        getattr(backend, "wants_full_batch", False)
        and not (sc and sc.get("enabled"))
        and not (dyn and dyn.get("enabled"))
    )

    precomputed = None

    if fast_batch:
        prompts: List[str] = []

        for doc_idx in range(len(test)):
            doc = (
                test.iloc[doc_idx]
                if hasattr(test, "iloc")
                else test[doc_idx]
            )

            sentences = D.get_source_sentences(doc)

            prompts.append(
                technique.build(
                    sentences,
                    TASK_ASPECT,
                    ctx,
                )
            )
        start = time.perf_counter()

        precomputed = backend.generate_batch(
            prompts,
            system=system_prompt or None,
        )

        # Validate backend output size.
        if len(precomputed) != len(test):
            raise RuntimeError(
                "Backend returned an unexpected number of outputs: "
                f"expected {len(test)}, got {len(precomputed)}."
            )

        elapsed = time.perf_counter() - start

        mean_document_latency = elapsed / max(len(test), 1)
        latencies = [mean_document_latency] * len(test)
        
        # start = time.perf_counter()

        # precomputed = backend.generate_batch(
        #     prompts,
        #     system=system_prompt or None,
        # )

        # elapsed = time.perf_counter() - start

        # mean_document_latency = elapsed / max(len(test), 1)
        # latencies = [mean_document_latency] * len(test)

    with partial_path.open("w", encoding="utf-8") as fh:
        for doc_idx in range(len(test)):
            doc = (
                test.iloc[doc_idx]
                if hasattr(test, "iloc")
                else test[doc_idx]
            )

            doc_id = D.get_doc_id(doc)
            sentences = D.get_source_sentences(doc)
            reference_summary = D.get_reference_summary(doc)
            silver_indices = D.gold_for_doc(doc)

            n_sentences = len(sentences)

            if precomputed is not None:
                raw_response = precomputed[doc_idx]
                pred_indices = _parse_prediction(
                    raw_response,
                    n_sentences,
                )

            else:
                prompt = technique.build(
                    sentences,
                    TASK_ASPECT,
                    ctx,
                )

                start = time.perf_counter()

                # Use the Track A wrapper infrastructure when self-consistency
                # or dynamic capping is enabled.
                if (
                    (sc and sc.get("enabled"))
                    or (dyn and dyn.get("enabled"))
                ):
                    caps = {TASK_ASPECT: cap_k}

                    predictions, raw_outputs = select_document(
                        backend=backend,
                        prompts=[prompt],
                        aspects=[TASK_ASPECT],
                        sentences=sentences,
                        caps=caps,
                        variant=variant,
                        sc=sc,
                        dyn=dyn,
                        system=system_prompt or None,
                    )

                    pred_indices = predictions.get(
                        TASK_ASPECT,
                        [],
                    )

                    raw_response = raw_outputs.get(
                        TASK_ASPECT,
                        "",
                    )

                else:
                    # raw_response = backend.generate(
                    #     prompt,
                    #     system=SYSTEM,
                    # )
                    raw_response = backend.generate_batch(
                        [prompt],
                        system=system_prompt or None,
                    )[0]
                    pred_indices = _parse_prediction(
                        raw_response,
                        n_sentences,
                    )

                latencies.append(
                    time.perf_counter() - start
                )

            # Enforce the fixed Track B cap only for capped variants.
            # if variant.cap is Cap.CAPPED:
            #     pred_indices = cap_indices_in_order(
            #         pred_indices,
            #         cap_k,
            #     )
            dynamic_cap_enabled = bool(
                dyn and dyn.get("enabled")
            )

            if (
                variant.cap is Cap.CAPPED
                and not dynamic_cap_enabled
            ):
                pred_indices = cap_indices_in_order(
                    pred_indices,
                    cap_k,
            )

            # Ensure predictions remain valid after wrappers/capping.
            # pred_indices = [
            #     int(index)
            #     for index in pred_indices
            #     if 1 <= int(index) <= n_sentences
            # ]

            # to ensure predictions remain valid after wrappers/capping.
            cleaned_indices: List[int] = []
            seen = set()

            for value in pred_indices:
                try:
                    index = int(value)
                except (TypeError, ValueError, OverflowError):
                    continue

                if not (1 <= index <= n_sentences):
                    continue

                if index in seen:
                    continue

                seen.add(index)
                cleaned_indices.append(index)

            pred_indices = cleaned_indices

            # Reconstruct the selected summaries once and save them in JSONL.
            pred_text = D.indices_to_text(
                sentences,
                pred_indices,
            )

            silver_text = D.indices_to_text(
                sentences,
                silver_indices,
            )

            exact_metrics = M.prf(
                silver_indices,
                pred_indices,
            )

            rouge_model: Dict[str, float] = {}
            rouge_oracle: Dict[str, float] = {}
            oracle_indices: List[int] = []

            if reference_summary:
                rouge_model = M.rouge(
                    reference_summary,
                    pred_text,
                )

                oracle_key = (
                    dataset_name,
                    doc_id,
                    "oracle",
                )

                if oracle_key not in oracle_cache:
                    oracle_cache[oracle_key] = M.build_oracle_indices(
                        sentences,
                        reference_summary,
                    )

                oracle_indices = oracle_cache[oracle_key]

                oracle_text = D.indices_to_text(
                    sentences,
                    oracle_indices,
                )

                rouge_oracle = M.rouge(
                    reference_summary,
                    oracle_text,
                )

            result = _row(
                doc_id=doc_id,
                doc_idx=doc_idx,
                dataset=dataset_name,
                n=n_sentences,
                pred=pred_indices,
                gold=silver_indices,
                pred_text=pred_text,
                silver_text=silver_text,
                reference_summary=reference_summary,
                metrics=exact_metrics,
                rouge_model=rouge_model,
                oracle_idx=oracle_indices,
                rouge_oracle=rouge_oracle,
                raw=raw_response,
            )

            fh.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Atomic replacement avoids leaving a complete-looking output after a
    # crashed or interrupted run.
    partial_path.replace(out_path)

    mean_latency = (
        sum(latencies) / len(latencies)
        if latencies
        else 0.0
    )

    metadata = {
        "variant": variant.slug,
        "dataset": dataset_name,
        "mean_latency_s": mean_latency,
        "n_docs": len(test),
        "cap_k": cap_k,
    }

    out_path.with_suffix(".meta.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def run(
    model_alias: str,
    experiment_yaml: str = "configs/experiment_xsum.yaml",
    models_yaml: str = "configs/models.yaml",
    grid_yaml: str = "configs/grid.yaml",
) -> None:
    """Run every configured prompt variant for one model and one dataset."""
    with open(experiment_yaml, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Configure the rationale cache directory from the experiment YAML.
    rationale_cfg = cfg.get("rationale", {})

    configure_rationale_cache(
        rationale_cfg.get(
            "cache_dir",
            "data/shots/rationales",
        )
    )

    seed = int(
        cfg.get("fewshot", {}).get("seed", 42)
    )

    inference_cfg = cfg.get("inference", {})
    self_consistency_cfg = cfg.get("self_consistency", {})
    dynamic_capper_cfg = cfg.get("dynamic_capper", {})

    # Self-consistency requires sampling. All ordinary runs remain
    # deterministic unless explicitly configured otherwise.
    if self_consistency_cfg.get("enabled"):
        temperature = float(
            self_consistency_cfg.get("temperature", 0.7)
        )
    else:
        temperature = float(
            inference_cfg.get("temperature", 0.0)
        )

    generation_config = GenConfig(
        max_new_tokens=int(
            inference_cfg.get("max_new_tokens", 512)
        ),
        temperature=temperature,
        seed=int(
            inference_cfg.get("seed", seed)
        ),
    )

    output_cfg = cfg.get("output", {})
    resume = bool(output_cfg.get("resume", True))
    results_dir = Path(
        output_cfg.get("results_dir", "results")
    )

    if "dataset" not in cfg:
        raise KeyError(
            f"'dataset' section is missing from {experiment_yaml}"
        )

    dataset_cfg = cfg["dataset"]

    dataset_name = str(dataset_cfg["name"])
    silver_path = dataset_cfg["silver_path"]

    test = D.load_split(silver_path)

    # Optional validation/train silver dataset used only for one-shot variants.
    train = None

    # if cfg.get("fewshot", {}).get("enabled", False):
    #     train_path = dataset_cfg.get("train_silver_path")

    #     if train_path:
    #         train = D.load_split(train_path)
    #     else:
    #         print(
    #             "[warning] Few-shot mode is enabled, but dataset."
    #             "train_silver_path is not configured. "
    #             "One-shot variants may fall back to no exemplar."
    #         )

    cap_k = _load_best_k(cfg)

    backend = get_backend(
        model_alias,
        models_yaml,
        generation_config,
    )

    variants = resolve_variants(grid_yaml)

    train = None

    fewshot_cfg = cfg.get("fewshot", {})
    fewshot_enabled = bool(
        fewshot_cfg.get("enabled", False)
    )

    has_one_shot_variants = any(
        variant.shot is Shot.ONE
        for variant in variants
    )

    if has_one_shot_variants:
        if not fewshot_enabled:
            raise RuntimeError(
                "The resolved grid contains one-shot variants, but "
                "fewshot.enabled is false in the experiment YAML."
            )

        train_path = dataset_cfg.get(
            "train_silver_path"
        )

        if not train_path:
            raise RuntimeError(
                "The resolved grid contains one-shot variants, but "
                "dataset.train_silver_path is missing."
            )

        train_path = Path(train_path)

        if not train_path.exists():
            raise FileNotFoundError(
                f"One-shot validation silver dataset not found: {train_path}"
            )

        train = D.load_split(train_path)

        if len(train) == 0:
            raise ValueError(
                f"One-shot validation dataset is empty: {train_path}"
            )

    output_dir = (
        results_dir
        / dataset_name
        / model_alias
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    oracle_cache: Dict = {}

    for variant in variants:
        out_path = output_dir / f"{variant.slug}.jsonl"

        if resume and out_path.exists():
            print(
                f"[skip] {dataset_name}/"
                f"{model_alias}/{variant.slug} "
                "(output already exists)"
            )
            continue

        print(
            f"[run]  {dataset_name}/"
            f"{model_alias}/{variant.slug}"
        )

        run_variant(
            variant=variant,
            backend=backend,
            test=test,
            cap_k=cap_k,
            seed=seed,
            oracle_cache=oracle_cache,
            out_path=out_path,
            dataset_name=dataset_name,
            train=train,
            sc=self_consistency_cfg,
            dyn=dynamic_capper_cfg,
        )

        # Release cached GPU memory between prompt variants.
        backend.free_memory()