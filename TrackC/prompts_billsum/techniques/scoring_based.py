"""scoring_based — BillSum sentence scoring with controlled Track C ablations.

Variants:
- scoring_based:
    full BillSum scoring prompt;
- scoring_no_length:
    removes only the soft conciseness/length guidance;
- scoring_no_redundancy:
    removes only the anti-redundancy guidance;
- scoring_based_trace:
    cached rationale demonstration for the full prompt.
"""

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import reasoning_example_block, render


_NAME = "scoring_based"


_SCORE_INTRODUCTION = (
    "Evaluate each sentence independently for its contribution to a concise "
    "summary of the whole bill.\n"
    "Assign one score from 1 to 5:\n"
    "   - 1: boilerplate, procedural material, minor detail, or information with "
    "little value for understanding the bill\n"
    "   - 2: weak supporting information with limited summary value\n"
    "   - 3: moderately useful information that provides context or secondary detail\n"
    "   - 4: important information about the bill's purpose, provisions, requirements, "
    "affected parties, conditions, funding, authority, eligibility, deadlines, "
    "exceptions, or effects\n"
    "   - 5: essential information that is central to understanding the bill as a whole\n"
    "After scoring:\n"
)

_SELECT_HIGH_SCORES = (
    "1. Select all sentences scoring 5.\n"
    "2. Include sentences scoring 4 when they contribute important information "
    "that is not already covered.\n"
)

_ANTI_REDUNDANCY = (
    "3. Avoid selecting redundant sentences even if they receive high scores; "
    "when two sentences convey substantially the same information, prefer the "
    "clearer or more complete one.\n"
)

_SOFT_LENGTH_GUIDANCE = (
    "4. Ensure the final sentence set covers the bill's main substantive information "
    "while remaining concise.\n"
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


# Full BillSum scoring prompt.
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

# Remove only the conciseness / length guidance.
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

# Remove only the anti-redundancy guidance.
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