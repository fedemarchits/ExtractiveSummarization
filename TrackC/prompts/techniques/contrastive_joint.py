"""contrastive_joint — contrastively classify sentence importance for
whole-document extractive summarization.

Each sentence is assigned exactly one category before the final selection.
The contrastive decision helps distinguish genuinely summary-worthy
information from supporting or redundant content.
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
    """Build the Track C contrastive prompt.

    ``task`` is retained only for compatibility with the common prompting
    interface. Track C always uses whole-document summarization.
    """
    del task

    return render(
        "summary",
        sentences,
        ctx,
        instructions=(
            "Classify each sentence before making the final selection.\n\n"
            "Assign exactly ONE label to every sentence:\n"
            "- core_information: contains information essential for understanding "
            "the document and should normally appear in the summary.\n"
            "- supporting_information: useful context or supporting details, but "
            "not essential if the core information is already covered.\n"
            "- redundant_information: repeats information expressed more clearly "
            "by another sentence.\n"
            "- irrelevant_information: background details, minor facts, quotations, "
            "examples, or other information that is not necessary for a concise "
            "summary.\n\n"
            "After classifying all sentences:\n"
            "1. Select the sentences labelled "
            "\"core_information\".\n"
            "2. Include a "
            "\"supporting_information\" sentence only if it provides important "
            "context that is not already covered.\n"
            "3. Never select sentences labelled "
            "\"redundant_information\" or "
            "\"irrelevant_information\".\n"
            "4. Ensure the final selection is concise, covers the document's main "
            "information, and avoids unnecessary repetition.\n"
            "Do not include the intermediate labels in the final output."
        ),
    )


register(
    Technique(
        name="contrastive_joint",
        build=build,
    )
)

