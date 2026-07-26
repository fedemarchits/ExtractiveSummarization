"""Resolve configs/grid.yaml + the technique registry into active Variants.

grid.yaml per technique:
  <name>: all                          -> the technique's full shot x cap grid
  <name>: [{shot: .., cap: ..}, ...]   -> only those cells
  (absent)                             -> default to the full grid
Requested cells are always intersected with what the technique declares valid
(e.g. negative_aware is one-shot only), so an over-broad grid can't create an
invalid variant.
"""
from __future__ import annotations

from typing import Dict, List

import yaml

from prompts.base import Cap, Shot, Variant, expand
from prompts.registry import all_techniques


def _valid_cells(tech) -> set:
    return {(v.shot, v.cap) for v in expand(tech)}


def resolve_variants(grid_yaml: str) -> List[Variant]:
    spec: Dict = (yaml.safe_load(open(grid_yaml)) or {}).get("techniques", {})
    out: List[Variant] = []
    for name, tech in all_techniques().items():
        valid = _valid_cells(tech)
        entry = spec.get(name, "all")
        if entry == "all" or entry is None:
            cells = valid
        else:
            cells = set()
            for item in entry:
                s, c = Shot(item["shot"]), Cap(item["cap"])
                if (s, c) in valid:
                    cells.add((s, c))
        out.extend(Variant(name, s, c) for (s, c) in valid if (s, c) in cells)
    return out
