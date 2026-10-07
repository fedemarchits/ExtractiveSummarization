"""Technique registry.

Each file under techniques/ calls @register on its Technique. Importing this
module discovers them all. Lookups are by technique name; variant expansion
lives in base.expand.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, List

from .base import Technique, Variant, expand

_REGISTRY: Dict[str, Technique] = {}


def register(tech: Technique) -> Technique:
    if tech.name in _REGISTRY:
        raise ValueError(f"duplicate technique: {tech.name}")
    _REGISTRY[tech.name] = tech
    return tech


def _discover() -> None:
    from . import techniques
    for mod in pkgutil.iter_modules(techniques.__path__):
        importlib.import_module(f"{techniques.__name__}.{mod.name}")


def all_techniques() -> Dict[str, Technique]:
    if not _REGISTRY:
        _discover()
    return dict(_REGISTRY)


def all_variants() -> List[Variant]:
    out: List[Variant] = []
    for t in all_techniques().values():
        out.extend(expand(t))
    return out
