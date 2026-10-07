"""contrastive_joint — contrast essential summary content with non-summary content."""

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
            "Classify each sentence internally before making the final selection.\n\n"
            "Assign exactly one category to each sentence:\n"
            "- essential: states information central to understanding the bill, such as its "
            "main purpose, key provisions, requirements, conditions, affected parties, "
            "funding or authority, eligibility, deadlines, exceptions, or major effects.\n"
            "- supporting: provides useful context or secondary detail that may be valuable "
            "when it adds important information not already covered by essential sentences.\n"
            "- redundant: repeats or substantially overlaps information expressed more clearly "
            "or completely by another sentence.\n"
            "- non-summary detail: contains procedural, boilerplate, administrative, narrowly "
            "technical, or minor information that is not necessary for a concise whole-bill summary.\n\n"
            "After classifying the sentences:\n"
            "1. Select the essential sentences.\n"
            "2. Select supporting sentences only when they add important information not already covered.\n"
            "3. Do not select redundant or non-summary-detail sentences.\n"
            "4. Keep the final selection concise, non-redundant, and representative of the bill's "
            "main substantive content.\n"
            "Do not include the intermediate classifications in the final output."
        ),
    )


register(
    Technique(
        name="contrastive_joint",
        build=build,
    )
)