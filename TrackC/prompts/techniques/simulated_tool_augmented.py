"""simulated_tool_augmented — use an imagined summary-analysis tool before
selecting sentences.

The tool is not actually executed. Instead, the model internally simulates its
output to encourage structured reasoning about sentence importance for
whole-document extractive summarization.
"""

from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import render


def build(
    sentences: Sequence[str],
    task: str,
    ctx: RenderCtx,
) -> str:
    """Build the Track C simulated tool-augmented prompt.

    ``task`` is retained for compatibility with the shared prompting
    interface. Track C should pass ``"summary"``.
    """
    del task

    return render(
        "summary",
        sentences,
        ctx,
        instructions=(
            "For each sentence, internally simulate the following tool:\n\n"
            "analyze_summary_value(sentence)\n\n"
            "The tool evaluates:\n"
            "- overall importance to the document,\n"
            "- contribution to summary coverage,\n"
            "- redundancy with other sentences,\n"
            "- whether the sentence mainly contains background information, "
            "minor details, quotations, or examples.\n\n"
            "The simulated tool returns one of:\n"
            "- essential\n"
            "- supporting\n"
            "- redundant\n"
            "- irrelevant\n\n"
            "Select all sentences classified as 'essential'.\n"
            "Include a 'supporting' sentence only if it provides important "
            "information that is not already covered.\n"
            "Never select sentences classified as 'redundant' or "
            "'irrelevant'.\n"
            "Return only the selected sentence indices."
        ),
    )


register(
    Technique(
        name="simulated_tool_augmented",
        build=build,
    )
)

