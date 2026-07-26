"""explanation_based — select only sentences that can be clearly justified
as necessary for a concise whole-document extractive summary.
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
    """Build the Track C explanation-based prompt.

    ``task`` is retained for compatibility with the shared Track A-style
    interface. Track C should pass ``"summary"``.
    """
    del task

    return render(
        "summary",
        sentences,
        ctx,
        instructions=(
            "Evaluate each candidate sentence by forming a brief internal "
            "justification for why it is needed in the final summary.\n"
            "1. A strong justification should show that the sentence contributes "
            "an essential event, claim, fact, cause, action, consequence, or result.\n"
            "2. Reject sentences whose justification depends only on minor detail, "
            "background information, repetition, examples, quotations, or local "
            "context that is not necessary for understanding the whole document.\n"
            "3. When two sentences convey the same information, keep only the one "
            "with the clearer or more complete summary contribution.\n"
            "4. Check that the selected set covers the document's main information "
            "without unnecessary repetition.\n"
            "5. Return only the indices of sentences with strong justifications.\n"
            "Do not include the justifications in the final output."
        ),
    )


register(
    Technique(
        name="explanation_based",
        build=build,
    )
)

