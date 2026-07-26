"""salience_inference — infer the document's central theme, score sentence
salience, and select the strongest whole-document summary content.

salience_inference:
    One-shot example is answer-only.

salience_inference_trace:
    One-shot example includes a cached reasoning trace.
"""

from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import reasoning_example_block, render


_NAME = "salience_inference"


def _instructions(task: str) -> str:
    """Return salience-inference instructions for Track C."""
    del task

    return (
        "1. First, infer the document's central topic, event, claim, or development "
        "in one sentence internally.\n"
        "2. Identify the main information a reader must retain to understand the "
        "whole document.\n"
        "3. Score each sentence internally from 0 to 2 for summary salience:\n"
        "   - 0: not useful for the final summary; minor, redundant, or background-only\n"
        "   - 1: somewhat useful supporting information, but not essential\n"
        "   - 2: essential and central information that should appear in the summary\n"
        "4. Select the sentences scoring 2.\n"
        "5. Include a sentence scoring 1 only when it adds important context not "
        "already covered by the selected sentences.\n"
        "6. Remove redundancy and preserve broad coverage of the document's main "
        "information.\n"
        "Do not include the inferred theme, scores, or reasoning in the final output."
    )


def _build(
    sentences: Sequence[str],
    task: str,
    ctx: RenderCtx,
    use_trace: bool,
) -> str:
    """Build the ordinary or trace-assisted salience prompt."""
    example_override = (
        reasoning_example_block(
            _NAME,
            task,
            ctx,
        )
        if use_trace
        else ""
    )

    return render(
        task,
        sentences,
        ctx,
        instructions=_instructions(task),
        example_override=example_override,
    )


register(
    Technique(
        name=_NAME,
        build=lambda sentences, task, ctx: _build(
            sentences,
            task,
            ctx,
            False,
        ),
    )
)

register(
    Technique(
        name=f"{_NAME}_trace",
        shots=(Shot.ONE,),
        build=lambda sentences, task, ctx: _build(
            sentences,
            task,
            ctx,
            True,
        ),
    )
)