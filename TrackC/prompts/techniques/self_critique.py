"""self_critique — draft candidates, critique them, and output a pruned
whole-document extractive summary.

Zero-shot:
    Draft and self-critique in one pass.

One-shot:
    The example demonstrates a draft list being pruned to the final silver
    selection, teaching the refinement operation.

The technique is intended to reduce over-selection, redundancy, keyword-only
matches, and low-value background details.
"""

from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import numbered, render


_INSTRUCTIONS = (
    "1. Draft a candidate list of sentences that appear useful for summarizing "
    "the whole document.\n"
    "2. Critique the draft sentence by sentence. Remove any candidate that:\n"
    "   - contains only background or minor detail,\n"
    "   - repeats information already covered more clearly elsewhere,\n"
    "   - matches the topic only through surface keywords,\n"
    "   - is an unnecessary quotation, example, or local detail,\n"
    "   - does not contribute an essential event, claim, fact, cause, action, "
    "consequence, or result.\n"
    "3. Check whether the remaining set covers the document's central information "
    "without unnecessary repetition.\n"
    "4. Output the pruned final selection only.\n"
    "Do not include the draft or critique in the final output."
)


def _refinement_example(ex) -> str:
    """Build a one-shot refinement example from a Track C silver exemplar."""
    gold = list(ex.gold_indices)
    n_sentences = len(ex.sentences)

    # Add one non-gold sentence to demonstrate pruning.
    spurious = next(
        (
            index
            for index in range(1, n_sentences + 1)
            if index not in gold
        ),
        None,
    )

    draft = sorted(
        set(
            gold
            + ([spurious] if spurious is not None else [])
        )
    )

    return (
        "Example (one-shot; exemplar from the validation silver split):\n"
        "Input document:\n"
        + numbered(ex.sentences)
        + "\n\n"
        f"Draft candidates: {draft}\n"
        "Critique: remove candidates that are redundant, background-only, "
        "minor, keyword-matched without real summary value, or otherwise "
        "unnecessary for a concise whole-document summary.\n"
        f"Final selection: {gold}\n\n"
        "---\n\n"
        "New document:\n"
    )


def build(
    sentences: Sequence[str],
    task: str,
    ctx: RenderCtx,
) -> str:
    """Build the Track C self-critique prompt.

    ``task`` is retained only for compatibility with the common interface.
    Track C should pass ``"summary"``.
    """
    del task

    example_override = ""

    if (
        ctx.shot is Shot.ONE
        and ctx.exemplar is not None
    ):
        example_override = _refinement_example(
            ctx.exemplar
        )

    return render(
        "summary",
        sentences,
        ctx,
        instructions=_INSTRUCTIONS,
        example_override=example_override,
    )


register(
    Technique(
        name="self_critique",
        build=build,
    )
)

