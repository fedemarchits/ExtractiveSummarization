"""Prompt-package selection helpers for Track C."""
from __future__ import annotations

import importlib
import sys


def activate_billsum_prompts() -> None:
    """Expose ``prompts_billsum`` under the existing ``prompts`` import names.

    Track C's engine intentionally keeps its original imports unchanged. This
    process-local mapping lets a second prompt package reuse the same runner
    without modifying or overwriting ``prompts/``.
    """
    package = importlib.import_module("prompts_billsum")
    sys.modules["prompts"] = package

    # Map shared modules explicitly so class/registry identity is consistent.
    for name in (
        "base",
        "rationale",
        "shared",
        "fewshot",
        "centrality",
        "wrappers",
        "registry",
    ):
        module = importlib.import_module(f"prompts_billsum.{name}")
        sys.modules[f"prompts.{name}"] = module
