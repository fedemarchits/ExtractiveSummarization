"""tool_augmented — use real TF-IDF centrality as supporting evidence, with controlled ablations."""

from typing import Sequence

from ..base import RenderCtx, Technique
from ..centrality import numbered_with_centrality
from ..registry import register
from ..shared import numbered, render


_PREAMBLE = (
    "Each sentence is annotated with a TF-IDF centrality score. Higher scores indicate that the "
    "sentence is more lexically central to the bill as a whole. Treat this score as supporting "
    "evidence only; it is not a direct measure of legal or substantive importance."
)

_NEUTRAL_HEADER = (
    "Select source sentences that together form a concise, informative, and non-redundant "
    "summary of the whole bill."
)

_INSTRUCTIONS = (
    "1. Consider both the substantive content of each sentence and its TF-IDF centrality score.\n"
    "2. Prioritize sentences that capture the bill's main purpose, major actions, requirements, "
    "affected parties, scope, funding, eligibility, deadlines, exceptions, or other important "
    "substantive effects.\n"
    "3. Use higher centrality as supporting evidence when choosing between otherwise similarly "
    "informative sentences.\n"
    "4. Do not rely on centrality alone; prefer a lower-centrality sentence when it contributes "
    "important information that is not covered elsewhere.\n"
    "5. Avoid selecting multiple sentences that express substantially the same information, "
    "even when they have high centrality scores.\n"
    "6. Avoid boilerplate, procedural, or overly detailed selections, and ensure broad coverage "
    "of the bill's most important provisions while remaining concise.\n"
    "7. Return only the selected sentence indices."
)


def _build(
    sentences: Sequence[str],
    task: str,
    ctx: RenderCtx,
    *,
    use_metadata: bool,
    use_roleplay: bool,
) -> str:
    if use_metadata:
        preamble = _PREAMBLE
        render_fn = numbered_with_centrality
        instructions = _INSTRUCTIONS
    else:
        preamble = ""
        render_fn = numbered

        # Same selection logic, with only the TF-IDF metadata and references
        # to centrality removed.
        instructions = (
            "1. Consider the substantive content of every sentence.\n"
            "2. Prioritize sentences that capture the bill's main purpose, major actions, "
            "requirements, affected parties, scope, funding, eligibility, deadlines, exceptions, "
            "or other important substantive effects.\n"
            "3. Prefer sentences that contribute important information not covered elsewhere.\n"
            "4. Avoid selecting multiple sentences that express substantially the same information.\n"
            "5. Avoid boilerplate, procedural, or overly detailed selections, and ensure broad "
            "coverage of the bill's most important provisions while remaining concise.\n"
            "6. Return only the selected sentence indices."
        )

    return render(
        task,
        sentences,
        ctx,
        preamble=preamble,
        render_fn=render_fn,
        instructions=instructions,
        header_override=None if use_roleplay else _NEUTRAL_HEADER,
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