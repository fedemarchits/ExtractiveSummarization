"""negative_aware — one-shot prompt with explicit near-miss negatives.

The one-shot example highlights sentences that appear important at first glance
but should NOT be selected because they contribute little to a concise
whole-document summary.

This reduces false positives by teaching the model to distinguish essential
summary content from background information, redundancy, quotations, minor
details, and other tempting but unnecessary sentences.

One-shot by design: the negative examples are the key idea, therefore there is
no zero-shot variant.
"""

from __future__ import annotations

from typing import Sequence

from ..base import Cap, RenderCtx, Shot, Technique
from ..registry import register
from ..shared import numbered, render


def _negative_example(ex) -> str:
    """Build the one-shot demonstration with explicit near-miss negatives."""
    gold = list(ex.gold_indices)
    n = len(ex.sentences)

    near_miss = [
        i
        for i in range(1, n + 1)
        if i not in gold
    ]

    near_miss_text = (
        ", ".join(map(str, near_miss))
        if near_miss
        else "none"
    )

    return (
        "Example (one-shot; exemplar from the training split):\n"
        "Input:\n"
        + numbered(ex.sentences)
        + "\n\n"
        f"Selected indices: {gold}\n"
        "Near-miss sentences NOT selected "
        "(they may contain interesting information but are omitted because "
        "they are redundant, background-only, minor details, quotations, "
        "examples, or otherwise unnecessary for a concise summary): "
        f"{near_miss_text}\n\n"
        "---\n\nNew task:\n"
    )


def build(
    sentences: Sequence[str],
    task: str,
    ctx: RenderCtx,
) -> str:
    """Build the Track C negative-aware prompt."""

    del task

    example_override = ""

    if (
        ctx.shot is Shot.ONE
        and ctx.exemplar is not None
    ):
        example_override = _negative_example(
            ctx.exemplar
        )

    return render(
        "summary",
        sentences,
        ctx,
        instructions=(
            "Select only sentences that are essential for a concise summary of "
            "the entire document.\n"
            "Exclude sentences that appear informative but are mainly:\n"
            "- background information,\n"
            "- redundant with another selected sentence,\n"
            "- quotations,\n"
            "- examples,\n"
            "- minor details,\n"
            "- local context that is unnecessary for understanding the document.\n"
            "Use the near-miss examples above as guidance when deciding which "
            "sentences should be excluded."
        ),
        example_override=example_override,
    )


register(
    Technique(
        name="negative_aware",
        build=build,
        shots=(Shot.ONE,),
        caps=(Cap.UNCAPPED, Cap.CAPPED),
        note="one-shot only; exemplar demonstrates near-miss summary negatives",
    )
)

