"""chain_of_thought — internally reason over summary value before selection."""
from typing import Sequence
from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import reasoning_example_block, render

_NAME = "chain_of_thought"

def _instructions(task: str) -> str:
    return (
        "1. Think step-by-step internally about the bill's main purpose and provisions.\n"
        "2. For each sentence, determine what important information it contributes.\n"
        "3. Select sentences containing essential, non-redundant information needed to "
        "understand the bill as a whole.\n"
        "4. Remove procedural boilerplate, minor details, and sentences whose information "
        "is already covered more clearly elsewhere.\n"
        "Do not include the reasoning in the final output."
    )

def _build(sentences: Sequence[str], task: str, ctx: RenderCtx, use_trace: bool) -> str:
    override = reasoning_example_block(_NAME, task, ctx) if use_trace else ""
    return render(task, sentences, ctx, instructions=_instructions(task), example_override=override)

register(Technique(name=_NAME, build=lambda s, a, c: _build(s, a, c, False)))
register(Technique(name=_NAME + "_trace", shots=(Shot.ONE,), build=lambda s, a, c: _build(s, a, c, True)))
