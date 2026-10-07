"""simulated_tool_augmented — simulate a bill summary-value analysis tool before selection."""

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import render


def build(
    sentences: Sequence[str],
    task: str,
    ctx: RenderCtx,
) -> str:
    return render(
        task,
        sentences,
        ctx,
        instructions=(
            "For each sentence, internally simulate the following tool:\n\n"
            "analyze_summary_value(sentence)\n\n"
            "The tool evaluates:\n"
            "- importance to understanding the bill as a whole,\n"
            "- contribution to coverage of the bill's purpose and major provisions,\n"
            "- whether the sentence adds unique substantive information,\n"
            "- redundancy with other sentences,\n"
            "- whether the sentence is mainly boilerplate, procedural language, "
            "minor implementation detail, or narrowly technical information.\n\n"
            "The simulated tool returns one of:\n"
            "- essential: central information that should normally appear in the summary\n"
            "- supporting: useful information that may be included if it adds important "
            "content not already covered\n"
            "- redundant: information substantially covered more clearly or completely elsewhere\n"
            "- non_summary_detail: information not necessary for a concise whole-bill summary\n\n"
            "Select all sentences classified as 'essential'.\n"
            "Include a 'supporting' sentence only if it contributes important information "
            "that is not already covered by the selected sentences.\n"
            "Never select sentences classified as 'redundant' or 'non_summary_detail'.\n"
            "Return only the selected sentence indices."
        ),
    )


register(
    Technique(
        name="simulated_tool_augmented",
        build=build,
    )
)