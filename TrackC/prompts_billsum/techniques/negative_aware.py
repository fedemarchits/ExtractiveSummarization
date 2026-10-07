"""negative_aware — one-shot selection with explicit hard-negative near misses."""
from typing import Sequence

from ..base import Cap, RenderCtx, Shot, Technique
from ..registry import register
from ..shared import numbered, render


def _negative_example(ex) -> str:
    selected = list(ex.gold_indices)
    near_miss = list(getattr(ex, "near_miss_indices", []))
    nm = ", ".join(map(str, near_miss)) if near_miss else "none"
    return (
        "Example (one-shot; BillSum silver training example):\n"
        "Input bill:\n" + numbered(ex.sentences) + "\n\n"
        f"Near-miss indices NOT selected: {nm}\n"
        "The near-miss sentences may appear summary-worthy, but they are "
        "redundant, overly detailed, procedural, boilerplate, or less central "
        "than the selected sentences.\n\n"
        # The demonstrated answer uses the required return format; a bare
        # "Selected indices: [...]" line is not parseable by the runner.
        "Correct output:\n"
        f'{{"selected_sentences": {selected}}}\n\n'
        "---\n\nNew bill:\n"
    )


def build(sentences: Sequence[str], task: str, ctx: RenderCtx) -> str:
    override = (
        _negative_example(ctx.exemplar)
        if ctx.shot is Shot.ONE and ctx.exemplar is not None
        else ""
    )
    return render(
        task,
        sentences,
        ctx,
        instructions=(
            "Select only sentences with strong value for a concise whole-bill summary. "
            "Prefer sentences that capture the bill's main purpose, major actions, "
            "requirements, affected parties, and important substantive effects. "
            "Explicitly reject plausible near-miss sentences that are topically related "
            "but redundant, boilerplate, procedural, overly specific, narrowly technical, "
            "or less important than another sentence covering the same information."
        ),
        example_override=override,
    )


register(
    Technique(
        name="negative_aware",
        build=build,
        shots=(Shot.ONE,),
        caps=(Cap.UNCAPPED, Cap.CAPPED),
        note="one-shot only; example includes explicit hard-negative near misses",
    )
)
