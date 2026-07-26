"""tool_augmented — sentence centrality annotations guide whole-document
extractive summarization, with controlled component ablations.

Variants:
- tool_augmented:
    original full prompt with TF-IDF metadata and role-play scaffolding;
- tool_no_meta:
    same prompt and role-play, but without centrality annotations;
- tool_no_roleplay:
    same prompt and centrality annotations, but without expert role-play.
"""

from __future__ import annotations

from typing import Sequence

from ..base import RenderCtx, Technique
from ..centrality import numbered_with_centrality
from ..registry import register
from ..shared import numbered, render


_PREAMBLE = (
    "Each sentence is annotated with a TF-IDF centrality score. "
    "Higher scores indicate that the sentence is more central to the document. "
    "Use the centrality scores as supporting evidence together with the sentence "
    "content when deciding which sentences belong in the final summary."
)

_NEUTRAL_HEADER = (
    "Select source sentences that together form a concise, informative, "
    "and non-redundant summary of the whole document."
)

_INSTRUCTIONS = (
    "1. Consider both the semantic content and the TF-IDF centrality score "
    "of every sentence.\n"
    "2. Prioritize sentences that contain essential information about the "
    "document while also having relatively high centrality.\n"
    "3. Do not rely on centrality alone—prefer a lower-centrality sentence "
    "if it contains unique information that is important for the summary.\n"
    "4. Avoid selecting multiple high-centrality sentences that express "
    "essentially the same information.\n"
    "5. Ensure the final selection provides broad coverage of the document's "
    "main information while remaining concise.\n"
    "6. Return only the selected sentence indices."
)


def _build(
    sentences: Sequence[str],
    task: str,
    ctx: RenderCtx,
    *,
    use_metadata: bool,
    use_roleplay: bool,
) -> str:
    """Build the full or ablated tool-augmented prompt."""
    del task

    if use_metadata:
        preamble = _PREAMBLE
        render_fn = numbered_with_centrality
        instructions = _INSTRUCTIONS
    else:
        preamble = ""
        render_fn = numbered

        # Keep the original logic as closely as possible, but remove references
        # to centrality because no centrality values are shown in this ablation.
        instructions = (
            "1. Consider the semantic content of every sentence.\n"
            "2. Prioritize sentences that contain essential information about "
            "the document.\n"
            "3. Prefer a sentence if it contains unique information that is "
            "important for the summary.\n"
            "4. Avoid selecting multiple sentences that express essentially "
            "the same information.\n"
            "5. Ensure the final selection provides broad coverage of the "
            "document's main information while remaining concise.\n"
            "6. Return only the selected sentence indices."
        )

    return render(
        "summary",
        sentences,
        ctx,
        preamble=preamble,
        render_fn=render_fn,
        instructions=instructions,
        header_override=(
            None
            if use_roleplay
            else _NEUTRAL_HEADER
        ),
    )


register(
    Technique(
        name="tool_augmented",
        build=lambda sentences, task, ctx: _build(
            sentences,
            task,
            ctx,
            use_metadata=True,
            use_roleplay=True,
        ),
    )
)

register(
    Technique(
        name="tool_no_meta",
        build=lambda sentences, task, ctx: _build(
            sentences,
            task,
            ctx,
            use_metadata=False,
            use_roleplay=True,
        ),
    )
)

register(
    Technique(
        name="tool_no_roleplay",
        build=lambda sentences, task, ctx: _build(
            sentences,
            task,
            ctx,
            use_metadata=True,
            use_roleplay=False,
        ),
        system_override="",
    )
)