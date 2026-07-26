"""Prompt technique base types and variant expansion.

A *technique* defines the core instruction. A *variant* is one of the four
combinations (zero/one-shot x capped/uncapped) derived mechanically from it,
so coverage is total by construction instead of copy-pasted per case.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Sequence



class Shot(str, Enum):
    ZERO = "zero_shot"
    ONE = "one_shot"


class Cap(str, Enum):
    UNCAPPED = "uncapped"
    CAPPED = "capped"


@dataclass
class Exemplar:
    """One few-shot exemplar. Gold indices are 1-based. Drawn from TRAIN only."""
    sentences: Sequence[str]
    gold_indices: Sequence[int]


# A builder turns (sentences, aspect, ctx) into the final user prompt string.
Builder = Callable[[Sequence[str], str, "RenderCtx"], str]


@dataclass
class RenderCtx:
    """Everything a builder may need beyond the raw sentences."""
    shot: Shot
    cap: Cap
    k: Optional[int] = None            # cap value, only when cap == CAPPED
    exemplar: Optional[Exemplar] = None  # only when shot == ONE


# @dataclass
# class Technique:
#     name: str
#     build: Builder
#     # Some techniques only make sense in a subset (e.g. negative_aware is
#     # one-shot only). Restrict here; expansion respects it.
#     shots: Sequence[Shot] = (Shot.ZERO, Shot.ONE)
#     caps: Sequence[Cap] = (Cap.UNCAPPED, Cap.CAPPED)
#     note: str = ""
@dataclass(frozen=True)
class Technique:
    name: str
    build: Builder
    shots: Sequence[Shot] = (
        Shot.ZERO,
        Shot.ONE,
    )
    caps: Sequence[Cap] = (
        Cap.UNCAPPED,
        Cap.CAPPED,
    )
    note: str = ""

    # None means use runner.py's default system prompt.
    # An empty string means send no system prompt.
    system_override: Optional[str] = None


@dataclass(frozen=True)
class Variant:
    technique: str
    shot: Shot
    cap: Cap

    @property
    def slug(self) -> str:
        return f"{self.technique}__{self.shot.value}__{self.cap.value}"


def expand(tech: Technique) -> List[Variant]:
    """All valid (shot, cap) variants for a technique."""
    return [
        Variant(tech.name, s, c)
        for s in tech.shots
        for c in tech.caps
    ]
