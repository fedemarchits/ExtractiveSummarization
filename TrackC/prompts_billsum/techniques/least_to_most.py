"""least_to_most — build the whole-bill extractive summary from simple decisions to final selection."""

from typing import Sequence

from ..base import RenderCtx, Technique
from ..registry import register
from ..shared import render


def build(sentences: Sequence[str], task: str, ctx: RenderCtx) -> str:
    return render(
        task,
        sentences,
        ctx,
        instructions=(
            "Solve the summarization task from simpler decisions to the final sentence selection:\n"
            "1. Identify the bill's overall purpose, principal action, or main policy objective.\n"
            "2. Break the bill into its main substantive content units, such as major provisions, "
            "requirements, affected parties, authorities, conditions, funding, eligibility, "
            "deadlines, exceptions, and consequences.\n"
            "3. For each important content unit, identify the sentence that expresses it most "
            "clearly and completely.\n"
            "4. Add supporting sentences only when they contribute important information not "
            "already covered by the selected sentences.\n"
            "5. Remove sentences that are redundant, procedural, boilerplate, overly detailed, "
            "or less informative than another selected sentence.\n"
            "6. Check that the final set gives a concise but sufficiently complete summary of "
            "the bill's main substantive content.\n"
            "7. Return only the selected sentence indices."
        ),
    )


register(
    Technique(
        name="least_to_most",
        build=build,
    )
)