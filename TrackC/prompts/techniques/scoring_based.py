"""scoring_based — assign a summary-importance score to each sentence and
select the highest-scoring ones, with controlled prompt ablations.

Variants:
- scoring_based:
    original full scoring prompt;
- scoring_no_length:
    removes only the soft conciseness/length guidance;
- scoring_no_redundancy:
    removes only the anti-redundancy guidance;
- scoring_based_trace:
    cached rationale demonstration for the full prompt.
"""

from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import reasoning_example_block, render


_NAME = "scoring_based"


_SCORE_INTRODUCTION = (
    "Evaluate each sentence independently for its contribution to a concise "
    "summary of the whole document.\n"
    "Assign one score from 1 to 5:\n"
    "   - 1: irrelevant, background information, or minor detail\n"
    "   - 2: weak supporting information with little summary value\n"
    "   - 3: moderately useful information that may provide context\n"
    "   - 4: important information that should normally appear in the summary\n"
    "   - 5: essential information that is central to understanding the document\n"
    "After scoring:\n"
)

_SELECT_HIGH_SCORES = (
    "1. Select all sentences scoring 5.\n"
    "2. Include sentences scoring 4 when they contribute important information "
    "that is not already covered.\n"
)

_ANTI_REDUNDANCY = (
    "3. Avoid selecting redundant sentences even if they receive high scores.\n"
)

_SOFT_LENGTH_GUIDANCE = (
    "4. Ensure the final sentence set provides broad coverage of the document's "
    "main information while remaining concise.\n"
)

_FINAL_RULE = (
    "Do not include the scores or reasoning in the final output."
)


def _instructions(
    task: str,
    *,
    use_length: bool,
    use_redundancy: bool,
) -> str:
    """Return the full or ablated Track C scoring instructions."""
    del task

    parts = [
        _SCORE_INTRODUCTION,
        _SELECT_HIGH_SCORES,
    ]

    if use_redundancy:
        parts.append(_ANTI_REDUNDANCY)

    if use_length:
        parts.append(_SOFT_LENGTH_GUIDANCE)

    parts.append(_FINAL_RULE)

    return "".join(parts)


def _build(
    sentences: Sequence[str],
    task: str,
    ctx: RenderCtx,
    *,
    use_length: bool,
    use_redundancy: bool,
    use_trace: bool,
) -> str:
    """Build the full, ablated, or trace-assisted scoring prompt."""
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
        instructions=_instructions(
            task,
            use_length=use_length,
            use_redundancy=use_redundancy,
        ),
        example_override=example_override,
    )


# Full original prompt.
register(
    Technique(
        name=_NAME,
        build=lambda sentences, task, ctx: _build(
            sentences,
            task,
            ctx,
            use_length=True,
            use_redundancy=True,
            use_trace=False,
        ),
    )
)

# Remove only:
# "Ensure the final sentence set ... while remaining concise."
register(
    Technique(
        name="scoring_no_length",
        build=lambda sentences, task, ctx: _build(
            sentences,
            task,
            ctx,
            use_length=False,
            use_redundancy=True,
            use_trace=False,
        ),
    )
)

# Remove only:
# "Avoid selecting redundant sentences even if they receive high scores."
register(
    Technique(
        name="scoring_no_redundancy",
        build=lambda sentences, task, ctx: _build(
            sentences,
            task,
            ctx,
            use_length=True,
            use_redundancy=False,
            use_trace=False,
        ),
    )
)

# Trace variant uses the unchanged full scoring prompt.
register(
    Technique(
        name=f"{_NAME}_trace",
        shots=(Shot.ONE,),
        build=lambda sentences, task, ctx: _build(
            sentences,
            task,
            ctx,
            use_length=True,
            use_redundancy=True,
            use_trace=True,
        ),
    )
)