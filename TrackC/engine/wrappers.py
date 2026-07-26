"""Orthogonal wrapper mechanisms, applied on top of any technique x variant.

self_consistency : sample the same prompt N times (temperature > 0) and keep
                   indices chosen by majority. Reduces unstable selections.
dynamic_llm_capper: a second pass where the model prunes its own selection to
                   the essential sentences (no fixed K). Alternative to a fixed
                   cap; when enabled it replaces the CAPPED post-hoc cap.

Both are off by default (see configs/experiment.yaml) and are the only place
the runner deviates from a single greedy pass.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Sequence

#from prompts.base import Cap
from prompts.wrappers import dynamic_capper_prompt
from .postprocess import clean_indices, safe_extract_json

#from .postprocess import cap_indices_in_order, clean_indices, safe_extract_json


def majority_vote(
    samples: Sequence[Sequence[int]], n_samples: int, threshold: Optional[int] = None
) -> List[int]:
    """Keep indices present in at least `threshold` samples (default > half)."""
    thr = threshold if threshold is not None else (n_samples // 2 + 1)
    counts: Counter = Counter()
    for s in samples:
        counts.update(set(s))
    return sorted(i for i, c in counts.items() if c >= thr)


def _parse_batch(outs: Sequence[str], aspects: Sequence[str], n: int) -> Dict[str, List[int]]:
    return {
        a: clean_indices(safe_extract_json(o).get("selected_sentences", []), n)
        for a, o in zip(aspects, outs)
    }


def dynamic_cap(backend, sentences, aspect, selected, system) -> List[int]:
    """Prune `selected` via a model pass; never adds indices. Falls back to the
    original selection if the prune pass yields nothing valid."""
    if not selected:
        return []
    out = backend.generate_batch(
        [dynamic_capper_prompt(sentences, aspect, selected)], system=system
    )[0]
    pruned = clean_indices(safe_extract_json(out).get("selected_sentences", []), len(sentences))
    keep = [i for i in pruned if i in set(selected)]
    return keep or list(selected)


def select_document(
    backend, prompts, aspects, sentences, caps, variant, sc, dyn, system
):
    """Produce per-aspect predictions for one doc, honoring the wrappers.

    Returns (preds, raws): preds maps aspect -> indices; raws maps aspect ->
    the raw model text (a list of N strings when self_consistency is on).
    """
    n = len(sentences)
    if sc and sc.get("enabled"):
        ns = int(sc.get("n_samples", 5))
        passes = [backend.generate_batch(prompts, system=system) for _ in range(ns)]
        samples = [_parse_batch(outs, aspects, n) for outs in passes]
        preds = {a: majority_vote([s[a] for s in samples], ns) for a in aspects}
        raws = {a: [outs[j] for outs in passes] for j, a in enumerate(aspects)}
    else:
        outs = backend.generate_batch(prompts, system=system)
        preds = _parse_batch(outs, aspects, n)
        raws = {a: outs[j] for j, a in enumerate(aspects)}

    for a in aspects:
        if dyn and dyn.get("enabled"):
            preds[a] = dynamic_cap(
                backend,
                sentences,
                a,
                preds[a],
                system,
            )

    return preds, raws

    # for a in aspects:
    #     if dyn and dyn.get("enabled"):
    #         preds[a] = dynamic_cap(backend, sentences, a, preds[a], system)
    #     elif variant.cap is Cap.CAPPED:
    #         preds[a] = cap_indices_in_order(preds[a], caps.get(a))
    # return preds, raws
