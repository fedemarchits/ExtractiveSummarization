"""salience_inference — infer the bill's central purpose and salient provisions,
then select sentences with the strongest whole-bill summary value.

salience_inference:
    One-shot example is answer-only.

salience_inference_trace:
    One-shot example includes a cached reasoning trace.
"""

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import reasoning_example_block, render


_NAME = "salience_inference"


def _instructions(task: str) -> str:
    del task

    return (
        "1. First, internally infer the bill's overall purpose, principal action, "
        "and most important substantive provisions.\n"
        "2. Identify the information a reader must retain to understand the bill as a whole.\n"
        "3. Score each sentence internally from 0 to 2 for summary salience:\n"
        "   - 0: not useful for the final summary; boilerplate, procedural, minor, "
        "redundant, or narrowly detailed\n"
        "   - 1: useful supporting information, but not essential on its own\n"
        "   - 2: central information about the bill's purpose, major provisions, "
        "requirements, affected parties, conditions, funding or authority, eligibility, "
        "deadlines, exceptions, or substantive effects\n"
        "4. Select the sentences scoring 2.\n"
        "5. Include a sentence scoring 1 only when it adds important information not "
        "already covered by the selected sentences.\n"
        "6. Remove redundancy and ensure broad coverage of the bill's main substantive content.\n"
        "Do not include the inferred themes, scores, or reasoning in the final output."
    )


def _build(
    sentences: Sequence[str],
    task: str,
    ctx: RenderCtx,
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