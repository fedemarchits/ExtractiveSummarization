"""least_to_most — decompose whole-document summarization into simpler steps."""

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
    """Build a least-to-most prompt for Track C.

    ``task`` is retained for compatibility with the Track A interface.
    Track C should pass ``"summary"``.
    """
    del task

    return render(
        "summary",
        sentences,
        ctx,
        instructions=(
            "Solve the summarization task from simpler decisions to the final "
            "sentence selection:\n"
            "1. Identify the document's overall topic, main event, or central claim.\n"
            "2. Divide the document's information into its main content units, such "
            "as key facts, causes, actions, evidence, and outcomes.\n"
            "3. For each content unit, identify the sentence that expresses it most "
            "clearly and completely.\n"
            "4. Remove sentences that are redundant, minor, background-only, or less "
            "informative than another selected sentence.\n"
            "5. Check that the final set gives a concise but sufficiently complete "
            "summary of the whole document.\n"
            "6. Return only the selected sentence indices."
        ),
    )


register(
    Technique(
        name="least_to_most",
        build=build,
    )
)

