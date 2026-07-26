"""Fail-fast preflight for a Track C run.

Checks the prerequisites that otherwise crash mid-run (after a model is already
loaded and GPU time is burning):

  1. silver test set exists for the dataset
  2. validation silver exists IF any one-shot variant is enabled
     (runner.py raises RuntimeError on a one-shot variant without it)
  3. a rationale cache exists IF any *_trace variant is enabled
     (prompts/shared.py raises FileNotFoundError otherwise)
  4. the requested model exists in the models YAML, and its backend's secrets
     are present (OPENAI_API_KEY for endpoint; HF token warned for local/vllm)

Exit code 0 = safe to run; 1 = blocking problem (with the exact fix printed);
warnings never block. Pure stdlib + pyyaml, so it runs before heavy deps load.

    python scripts/preflight.py --dataset xsum --model qwen35_9b \
        --models configs/models.vast.yaml [--grid configs/grid.yaml]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import yaml

EXPERIMENT_CONFIGS = {
    "xsum": "configs/experiment_xsum.yaml",
    "cnndm": "configs/experiment_cnndm.yaml",
}

# Techniques that require a prebuilt reasoning-trace cache. Kept explicit so the
# check is correct even if a _trace technique is ABSENT from grid.yaml (grid.py
# defaults absent techniques to their full grid, so they would still run).
TRACE_TECHNIQUES = (
    "chain_of_thought_trace",
    "self_ask_trace",
    "scoring_based_trace",
    "salience_inference_trace",
)


def _disabled(entry) -> bool:
    """A technique zeroed with an empty list is disabled."""
    return isinstance(entry, list) and len(entry) == 0


def _grid_flags(grid_path: str):
    """Return (one_shot_present, trace_present) from a grid.yaml.

    Mirrors engine/grid.py semantics: an ABSENT technique defaults to its full
    grid (so one-shot cells run), an empty list disables it.
    """
    spec = (yaml.safe_load(open(grid_path)) or {}).get("techniques", {})

    one_shot = False
    for name, entry in spec.items():
        if _disabled(entry):
            continue
        if entry in ("all", None):
            one_shot = True
        elif isinstance(entry, list):
            if any((cell or {}).get("shot") == "one_shot" for cell in entry):
                one_shot = True
    # Any technique not mentioned at all defaults to the full grid -> one-shot.
    # (We can't enumerate the registry here without heavy imports, but the
    # shipped grid lists every technique, so the loop above is authoritative.
    # If in doubt, one-shot is almost always present; assume True when empty.)
    if not spec:
        one_shot = True

    trace = any(not _disabled(spec.get(t, "all")) for t in TRACE_TECHNIQUES)
    return one_shot, trace


def _model_backend(models_path: str, alias: str):
    """Return the resolved model dict (yaml merges <<: anchors) or None."""
    cfg = yaml.safe_load(open(models_path))
    for m in cfg.get("models", []):
        if m.get("alias") == alias:
            return m
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(EXPERIMENT_CONFIGS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--models", default="configs/models.vast.yaml")
    ap.add_argument("--grid", default="configs/grid.yaml")
    args = ap.parse_args()

    errors: list[str] = []
    warns: list[str] = []

    exp = yaml.safe_load(open(EXPERIMENT_CONFIGS[args.dataset]))
    ds = exp.get("dataset", {})
    silver = ds.get("silver_path")
    val_silver = ds.get("train_silver_path")

    one_shot, trace = _grid_flags(args.grid)

    # 1. test silver
    if not silver or not Path(silver).exists():
        errors.append(
            f"missing test silver: {silver!r}\n"
            f"    fix: python generate_silver.py --dataset {args.dataset} [--max-docs N]"
        )

    # 2. validation silver (only if one-shot variants run)
    if one_shot:
        if not val_silver or not Path(val_silver).exists():
            errors.append(
                f"one-shot variants are enabled but validation silver is missing: {val_silver!r}\n"
                f"    (runner.py raises RuntimeError on one-shot without it)\n"
                f"    fix: python generate_silver.py --dataset {args.dataset} --split validation [--max-docs N]"
            )

    # 3. rationale cache (only if _trace variants run)
    if trace:
        cache_dir = (exp.get("rationale", {}) or {}).get(
            "cache_dir", "data/shots/rationales"
        )
        files = [p for p in glob.glob(f"{cache_dir}/**/*", recursive=True) if Path(p).is_file()]
        if not files:
            errors.append(
                f"_trace variants are enabled but the rationale cache is empty: {cache_dir!r}\n"
                f"    (prompts/shared.py raises FileNotFoundError otherwise)\n"
                f"    fix A: python -m scripts.build_rationales --dataset {args.dataset} --models {args.models}   (needs OPENAI_*)\n"
                f"    fix B: set every *_trace technique to [] in {args.grid} to skip them"
            )

    # 4. model + backend secrets
    m = _model_backend(args.models, args.model)
    if m is None:
        errors.append(f"model {args.model!r} not found in {args.models}")
    else:
        backend = m.get("backend", "local")
        if backend == "endpoint" and not os.environ.get("OPENAI_API_KEY"):
            errors.append(
                f"model {args.model!r} uses backend=endpoint but OPENAI_API_KEY is unset"
            )
        if backend in ("local", "vllm") and not (
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ):
            warns.append("HF_TOKEN unset — gated Gemma/Qwen weight downloads may fail")
        if backend == "local":
            warns.append(
                f"model {args.model!r} backend=local (HF, ~3-4 s/doc). For A100 speed "
                f"pass --models configs/models.vast.yaml (vllm)."
            )

    # report
    for w in warns:
        print(f"[preflight][warn] {w}")
    if errors:
        print("\n[preflight] BLOCKED — fix these before running:\n")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}\n")
        sys.exit(1)

    print(f"[preflight] OK — {args.model} / {args.dataset} ready to run.")


if __name__ == "__main__":
    main()
