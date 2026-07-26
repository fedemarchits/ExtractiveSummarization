"""Generate cached one-shot reasoning traces for Track C.

Examples:

    python -m scripts.build_rationales \
        --dataset xsum \
        --model qwen35_397b

    python -m scripts.build_rationales \
        --dataset cnndm \
        --model qwen35_397b

The script:

1. loads a validation silver dataset;
2. selects candidate exemplars deterministically;
3. asks one strong reference model to produce reasoning and indices;
4. compares the model indices against the silver indices;
5. keeps the first result above accept_f1, or the best result above min_f1;
6. saves one rationale JSON per reasoning technique.

The benchmark runner later reuses these cached demonstrations.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from engine import data as D
from engine.backends import GenConfig, get_backend
from engine.metrics import prf
#from engine.postprocess import extract_indices
from engine.postprocess import safe_extract_json
from prompts.fewshot import iter_exemplars
from prompts.rationale import (
    REASONING_TECHNIQUES,
    RationaleShot,
    configure_rationale_cache,
    save_rationale,
)


SYSTEM = (
    "You are an expert in extractive summarization. "
    "Provide concise reasoning and then a JSON answer."
)

TASK = "summary"


def _experiment_path(dataset: str) -> Path:
    mapping = {
        "xsum": Path("configs/experiment_xsum.yaml"),
        "cnndm": Path("configs/experiment_cnndm.yaml"),
    }

    try:
        return mapping[dataset]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset: {dataset}"
        ) from exc


def _load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment configuration not found: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(
            f"Experiment YAML must contain a mapping: {path}"
        )

    return config


def _numbered(sentences: Sequence[str]) -> str:
    return "\n".join(
        f"Sentence {index}: {sentence}"
        for index, sentence in enumerate(
            sentences,
            start=1,
        )
    )


def _reasoning_instructions(technique: str) -> str:
    instructions = {
        "chain_of_thought": (
            "Reason step by step about the document's central topic, key facts, "
            "coverage, redundancy, and which sentences are essential."
        ),
        "self_ask": (
            "For each sentence, ask whether removing it would cause the summary "
            "to lose important information, and whether the information is already "
            "covered elsewhere."
        ),
        "scoring_based": (
            "Assign each sentence an internal summary-importance score from 1 to 5. "
            "Explain which sentences are essential, useful, redundant, or minor."
        ),
        "salience_inference": (
            "Infer the document's central theme, assess sentence salience, and explain "
            "which sentences are necessary for a concise whole-document summary."
        ),
    }

    if technique not in instructions:
        raise ValueError(
            f"No rationale-generation instructions for technique: {technique}"
        )

    return instructions[technique]


def _build_prompt(
    technique: str,
    sentences: Sequence[str],
) -> str:
    return (
        f"{_reasoning_instructions(technique)}\n\n"
        "Document:\n"
        f"{_numbered(sentences)}\n\n"
        "Return your answer in exactly this structure:\n\n"
        "Reasoning:\n"
        "<your concise reasoning>\n\n"
        'Final JSON:\n'
        '{"selected_sentences": [1, 2, 3]}\n\n'
        "Use only valid 1-based sentence indices."
    )


def _parse_rationale(raw: str) -> str:
    text = str(raw or "").strip()

    match = re.search(
        r"Reasoning:\s*(.*?)(?:Final JSON:|"
        r'\{\s*"selected_sentences")',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        rationale = match.group(1).strip()
        if rationale:
            return rationale

    # Graceful fallback: remove a trailing JSON object if possible.
    rationale = re.sub(
        r'\{\s*"selected_sentences"\s*:\s*\[[^\]]*\]\s*\}\s*$',
        "",
        text,
        flags=re.DOTALL,
    ).strip()

    return rationale


def _clean_indices(
    values: Sequence[int],
    n_sentences: int,
) -> List[int]:
    cleaned: List[int] = []
    seen = set()

    for value in values:
        if isinstance(value, bool):
            continue

        try:
            index = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        # except (TypeError, ValueError):
        #     continue

        if (
            1 <= index <= n_sentences
            and index not in seen
        ):
            cleaned.append(index)
            seen.add(index)

    return cleaned


def _generate_candidate(
    backend,
    technique: str,
    exemplar,
) -> Optional[RationaleShot]:
    prompt = _build_prompt(
        technique,
        exemplar.sentences,
    )

    # raw = backend.generate_batch(
    #     [prompt],
    #     system=SYSTEM,
    # )[0]
    outputs = backend.generate_batch(
        [prompt],
        system=SYSTEM,
    )

    if len(outputs) != 1:
        raise RuntimeError(
            "Rationale generation expected exactly one backend output, "
            f"but received {len(outputs)}."
        )

    raw = outputs[0]

    # predicted = _clean_indices(
    #     extract_indices(raw),
    #     len(exemplar.sentences),
    # )
    parsed = safe_extract_json(raw)

    predicted = _clean_indices(
        parsed.get(
            "selected_sentences",
            [],
        ),
        len(exemplar.sentences),
    )

    if not predicted:
        return None

    rationale = _parse_rationale(raw)

    if not rationale:
        return None

    metrics = prf(
        exemplar.gold_indices,
        predicted,
    )

    return RationaleShot(
        technique=technique,
        aspect=TASK,
        source_model=backend.hf_id
        if hasattr(backend, "hf_id")
        else getattr(
            backend,
            "model_id",
            "reference_model",
        ),
        exemplar_sentences=list(
            exemplar.sentences
        ),
        shown_indices=predicted,
        gold_indices=list(
            exemplar.gold_indices
        ),
        f1=float(metrics["f1"]),
        rationale=rationale,
    )


def _build_one_rationale(
    backend,
    technique: str,
    exemplars,
    accept_f1: float,
    min_f1: float,
) -> Optional[RationaleShot]:
    best: Optional[RationaleShot] = None

    for exemplar_index, exemplar in enumerate(
        exemplars,
        start=1,
    ):
        print(
            f"[rationale] {technique}: "
            f"trying exemplar {exemplar_index}"
        )

        candidate = _generate_candidate(
            backend,
            technique,
            exemplar,
        )

        if candidate is None:
            continue

        print(
            f"[rationale] {technique}: "
            f"candidate F1={candidate.f1:.4f}"
        )

        if best is None or candidate.f1 > best.f1:
            best = candidate

        if candidate.f1 >= accept_f1:
            return candidate

    if best is not None and best.f1 >= min_f1:
        return best

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cached Track C rationale demonstrations."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["xsum", "cnndm"],
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Reference-model alias. Defaults to rationale.source_model "
            "from the experiment YAML."
        ),
    )

    parser.add_argument(
        "--models",
        default="configs/models.yaml",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    experiment_path = _experiment_path(
        args.dataset
    )
    config = _load_yaml(
        experiment_path
    )

    dataset_config = config.get(
        "dataset",
        {}
    )
    rationale_config = config.get(
        "rationale",
        {}
    )
    fewshot_config = config.get(
        "fewshot",
        {}
    )

    validation_path = dataset_config.get(
        "train_silver_path"
    )

    if not validation_path:
        raise RuntimeError(
            "dataset.train_silver_path is required to build rationales."
        )

    validation_path = Path(
        validation_path
    )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation silver dataset not found: {validation_path}"
        )

    cache_dir = Path(
        rationale_config.get(
            "cache_dir",
            f"data/shots/rationales/{args.dataset}",
        )
    )

    configure_rationale_cache(
        cache_dir
    )

    source_model = (
        args.model
        or rationale_config.get(
            "source_model"
        )
    )

    if not source_model:
        raise RuntimeError(
            "No rationale source model was configured."
        )

    max_tries = int(
        rationale_config.get(
            "max_exemplar_tries",
            10,
        )
    )
    accept_f1 = float(
        rationale_config.get(
            "accept_f1",
            0.7,
        )
    )
    min_f1 = float(
        rationale_config.get(
            "min_f1",
            0.4,
        )
    )
    max_new_tokens = int(
        rationale_config.get(
            "max_new_tokens",
            1024,
        )
    )
    seed = int(
        fewshot_config.get(
            "seed",
            42,
        )
    )

    if not 0.0 <= min_f1 <= accept_f1 <= 1.0:
        raise ValueError(
            "Expected 0 <= min_f1 <= accept_f1 <= 1."
        )

    validation_docs = D.load_split(
        validation_path
    )

    if len(validation_docs) == 0:
        raise ValueError(
            f"Validation silver dataset is empty: {validation_path}"
        )

    backend = get_backend(
        alias=source_model,
        models_yaml=args.models,
        gen=GenConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            seed=seed,
        ),
    )

    techniques = sorted(
        REASONING_TECHNIQUES
    )

    for technique in techniques:
        output_path = (
            cache_dir
            / f"{technique}__{TASK}.json"
        )

        if output_path.exists() and not args.overwrite:
            print(
                f"[skip] {output_path} already exists"
            )
            continue

        exemplars = list(
            iter_exemplars(
                validation_docs,
                task=TASK,
                seed=seed,
                limit=max_tries,
            )
        )

        if not exemplars:
            raise RuntimeError(
                f"No valid exemplars available for {technique}."
            )

        shot = _build_one_rationale(
            backend=backend,
            technique=technique,
            exemplars=exemplars,
            accept_f1=accept_f1,
            min_f1=min_f1,
        )

        if shot is None:
            print(
                f"[warning] No acceptable rationale found for "
                f"{technique}; no cache file written."
            )
            continue

        saved_path = save_rationale(
            shot,
            base_dir=cache_dir,
        )

        print(
            f"[written] {saved_path} "
            f"(F1={shot.f1:.4f})"
        )

    backend.free_memory()


if __name__ == "__main__":
    main()