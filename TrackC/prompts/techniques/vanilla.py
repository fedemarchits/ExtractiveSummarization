"""vanilla — direct extractive summarization baseline.

This prompt performs no explicit reasoning or intermediate steps. The model is
asked directly to select the sentences that best form a concise summary of the
entire document.
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
    """Build the Track C vanilla prompt.

    ``task`` is retained only for compatibility with the common prompting
    interface. Track C should pass ``"summary"``.
    """
    del task

    return render(
        "summary",
        sentences,
        ctx,
        instructions=(
            "Select the smallest set of sentences that together forms a concise "
            "and informative summary of the entire document.\n"
            "Choose sentences that contain the most important information while "
            "avoiding redundancy, minor details, quotations, and background "
            "information.\n"
            "Return only the selected sentence indices."
        ),
    )


register(
    Technique(
        name="vanilla",
        build=build,
    )
)