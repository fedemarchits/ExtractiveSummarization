"""Parse model output into selected indices, then clean/cap them.

Ported from the old pipeline's postprocess, unchanged in behavior:
robust JSON extraction + in-order cap. `clean_indices` folds the old inline
"sorted set of in-range ints" step into one reusable function.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence


def safe_extract_json(text: str) -> Dict[str, Any]:
    """Parse a response into {"selected_sentences": [...]}. {} on total failure.

    Order: direct parse -> strip md fences -> last {...} with the key ->
    bare array on its own line.
    """
    text = text.strip()

    try:
        js = json.loads(text)
        if isinstance(js, dict) and "selected_sentences" in js:
            return js
        if isinstance(js, list):
            return {"selected_sentences": js}
    except Exception:
        pass

    clean = re.sub(r"^```[\w-]*\s*\n", "", text, flags=re.S)
    clean = re.sub(r"\n```$", "", clean, flags=re.S).strip()

    for s in reversed(re.findall(r"\{[\s\S]*?\}", clean)):
        try:
            js = json.loads(s)
            if isinstance(js, dict) and "selected_sentences" in js:
                return js
        except Exception:
            continue

    m = re.search(r"(?m)^\s*\[(?:\s*\d+\s*(?:,\s*\d+\s*)*)?\]\s*$", clean)
    if m:
        try:
            return {"selected_sentences": json.loads(m.group(0))}
        except Exception:
            pass

    return {}


def clean_indices(raw: Sequence[Any], n_sentences: int) -> List[int]:
    """Sorted, unique, 1-based indices within [1, n_sentences]."""
    return sorted({
        int(v) for v in raw
        if isinstance(v, (int, float)) and 1 <= int(v) <= n_sentences
    })


def cap_indices_in_order(indices: List[int], cap: Optional[int]) -> List[int]:
    """Keep at most `cap` indices, preserving ascending order."""
    if cap is None or cap <= 0:
        return indices
    return indices[:cap]


def parse_selection(text: str, n_sentences: int, cap: Optional[int] = None) -> List[int]:
    """Full pipeline: extract JSON -> clean -> optional post-hoc cap."""
    raw = safe_extract_json(text).get("selected_sentences", [])
    idx = clean_indices(raw, n_sentences)
    return cap_indices_in_order(idx, cap)
