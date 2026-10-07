"""explanation_based — select sentences whose inclusion can be justified by summary value."""

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
            "Evaluate each candidate sentence by forming a brief internal justification "
            "for why it is needed in a concise whole-bill summary.\n"
            "1. A strong justification should show that the sentence contributes important "
            "information about the bill's purpose, major provisions, requirements, affected "
            "parties, conditions, funding or authority, eligibility, deadlines, exceptions, "
            "or substantive effects.\n"
            "2. Reject sentences whose justification depends only on boilerplate, procedural "
            "language, minor implementation detail, narrowly technical information, or "
            "background that is not necessary for understanding the bill as a whole.\n"
            "3. When two sentences convey substantially the same information, keep only the "
            "one with the clearer or more complete summary contribution.\n"
            "4. Select only sentences whose contribution is important, specific, and "
            "non-redundant.\n"
            "5. Check that the selected set covers the bill's main substantive content without "
            "unnecessary repetition.\n"
            "Do not include the internal justifications in the final output."
        ),
    )


register(
    Technique(
        name="explanation_based",
        build=build,
    )
)