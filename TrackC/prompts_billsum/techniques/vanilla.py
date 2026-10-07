"""vanilla — direct BillSum extractive summarization baseline."""

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
            "Select the smallest set of source sentences that together forms a concise "
            "and informative summary of the entire bill.\n"
            "Choose sentences that capture the bill's main purpose, key provisions or actions, "
            "important requirements or conditions, and major consequences or effects.\n"
            "Avoid redundancy, procedural boilerplate, and minor details.\n"
            "Return only the selected sentence indices."
        ),
    )


register(
    Technique(
        name="vanilla",
        build=build,
    )
)