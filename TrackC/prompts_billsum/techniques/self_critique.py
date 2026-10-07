"""self_critique — draft candidate summary sentences, critique, and prune."""

from typing import Sequence

from ..base import RenderCtx, Shot, Technique
from ..registry import register
from ..shared import numbered, render


_INSTRUCTIONS = (
    "1. Draft a candidate list of sentences that appear useful for summarizing "
    "the whole bill.\n"
    "2. Critique the draft sentence by sentence. Remove any candidate that:\n"
    "   - is mainly boilerplate, procedural language, or minor implementation detail,\n"
    "   - repeats information already covered more clearly or completely elsewhere,\n"
    "   - appears relevant mainly because it contains topic-related keywords but "
    "does not add important substantive information,\n"
    "   - is overly specific, narrowly technical, or otherwise unnecessary for a "
    "concise whole-bill summary,\n"
    "   - does not contribute important information about the bill's purpose, major "
    "provisions, requirements, affected parties, conditions, funding or authority, "
    "eligibility, deadlines, exceptions, or substantive effects.\n"
    "3. Check that the remaining set covers the bill's main purpose and key substantive "
    "provisions without unnecessary repetition.\n"
    "4. Output only the pruned final selection.\n"
    "Do not include the draft or critique in the final output."
)


def _refinement_example(ex) -> str:
    selected = list(ex.gold_indices)

    spurious = next(
        (
            i
            for i in range(1, len(ex.sentences) + 1)
            if i not in selected
        ),
        None,
    )

    draft = sorted(
        set(
            selected
            + ([spurious] if spurious is not None else [])
        )
    )

    return (
        "Example (one-shot; BillSum training example):\n"
        "Input bill:\n"
        + numbered(ex.sentences)
        + "\n\n"
        + f"Draft candidates: {draft}\n"
        + "Critique: remove candidates that are redundant, boilerplate, procedural, "
          "minor, overly specific, superficially relevant, or otherwise low in "
          "whole-bill summary value.\n"
        + "Correct output:\n"
        + f'{{"selected_sentences": {selected}}}\n\n'
        + "---\n\n"
        + "New bill:\n"
    )


def build(
    sentences: Sequence[str],
    task: str,
    ctx: RenderCtx,
) -> str:
    override = (
        _refinement_example(ctx.exemplar)
        if ctx.shot is Shot.ONE and ctx.exemplar is not None
        else ""
    )

    return render(
        task,
        sentences,
        ctx,
        instructions=_INSTRUCTIONS,
        example_override=override,
    )


register(
    Technique(
        name="self_critique",
        build=build,
    )
)