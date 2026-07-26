"""self_ask — internally ask whether each sentence is essential to the summary.

self_ask:
    One-shot example is answer-only.

self_ask_trace:
    One-shot example includes a cached reasoning trace.
"""

from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import reasoning_example_block, render


_NAME = "self_ask"


def _instructions(task: str) -> str:
    """Return self-ask instructions for whole-document summarization."""
    del task

    return (
        "For each sentence, ask the following questions internally:\n"
        '1. "Does this sentence contain information essential to understanding '
        'the whole document?"\n'
        '2. "Would removing this sentence cause the summary to lose an important '
        'event, claim, fact, cause, action, consequence, or result?"\n'
        '3. "Is this information already expressed more clearly by another '
        'sentence?"\n'
        '4. "Is this sentence mainly background, repetition, a minor detail, '
        'an example, or a quotation that is not necessary for the summary?"\n'
        'Answer "Yes" or "No" internally for whether the sentence should be '
        "selected.\n"
        "Select only the sentences whose final answer is Yes.\n"
        "Do not include the questions, answers, or reasoning in the final output."
    )


def _build(
    sentences: Sequence[str],
    task: str,
    ctx: RenderCtx,
    use_trace: bool,
) -> str:
    """Build the ordinary or trace-assisted self-ask prompt."""
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

