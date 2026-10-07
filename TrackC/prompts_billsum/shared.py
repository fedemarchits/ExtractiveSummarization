"""Shared rendering helpers for BillSum-oriented Track C prompts.

This mirrors ``prompts/shared.py`` and differs from it ONLY in domain wording
("document" -> "bill", legislation-specific task description). Prompt layout,
rules, cap wording, exemplar format and the required return format are kept
identical, so that comparing the two prompt sets on BillSum isolates the effect
of the domain wording.

The existing ``aspect`` parameter is retained for compatibility with Track A's
prompt interfaces. In Track C, callers should pass ``"summary"``.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from .base import Cap, RenderCtx, Shot
from .rationale import load_rationale


SUMMARY_TASK = "summary"

RETURN_FORMAT = '{"selected_sentences": [1, 3, 5]}'

SHARED_RULES = """- Select only sentences that appear in the provided bill.
- Use the original 1-based sentence numbers.
- Do not return duplicate indices.
- Preserve the source-document order of the selected sentences.
- If no sentence should be selected, return {"selected_sentences": []}.
- Return only one valid JSON object.
- Do not include explanations, reasoning, Markdown, or text outside the JSON object."""

Renderer = Callable[[Sequence[str]], str]


def numbered(sentences: Sequence[str]) -> str:
    """Render source sentences as a 1-based numbered list."""
    return "\n".join(
        f"Sentence {index}: {str(sentence).strip()}"
        for index, sentence in enumerate(sentences, start=1)
    )


def header(task: str = SUMMARY_TASK) -> str:
    """Return the shared Track C system-style task description.

    ``task`` remains an argument so Track A technique interfaces do not need to
    be redesigned. Track C should normally pass ``"summary"``.
    """
    normalized_task = str(task or SUMMARY_TASK).strip().lower()

    if normalized_task != SUMMARY_TASK:
        # Do not silently recreate the obsolete ACLSum aspect task.
        # This warning inside the prompt makes accidental misuse visible while
        # still allowing the experiment to run.
        task_note = (
            f'The supplied task label is "{task}", but this Track C experiment '
            "uses whole-document summarization."
        )
    else:
        task_note = ""

    base = (
        "You are an expert in extractive summarization of legislation. "
        "Select the source sentences that together form the best concise, "
        "informative, and non-redundant summary of the whole bill. "
        "The selected summary must preserve the bill's original wording "
        "and must not introduce information that is not present in the source."
    )

    if task_note:
        return f"{base}\n\nNote: {task_note}"

    return base


def cap_bullet(ctx: RenderCtx, task: str = SUMMARY_TASK) -> str:
    """Return the fixed sentence-cap rule for capped variants."""
    del task  # Retained only for compatibility with existing callers.

    if ctx.cap is Cap.CAPPED and ctx.k is not None:
        return (
            f"- Select at most {int(ctx.k)} sentences in total.\n"
            "- Select fewer sentences when fewer are sufficient.\n"
        )

    return ""


def example_block(
    ctx: RenderCtx,
    render_fn: Optional[Renderer] = None,
) -> str:
    """Render a Track C one-shot exemplar.

    The exemplar must come from a separate training or validation silver split,
    never from the evaluation documents.
    """
    if ctx.shot is not Shot.ONE or ctx.exemplar is None:
        return ""

    exemplar = ctx.exemplar
    renderer = render_fn or numbered

    return (
        "Example:\n"
        "Input bill:\n"
        f"{renderer(exemplar.sentences)}\n\n"
        "Correct output:\n"
        f'{{"selected_sentences": {list(exemplar.gold_indices)}}}\n\n'
        "---\n\n"
        "New bill:\n"
    )


def reasoning_example_block(
    technique: str,
    task: str,
    ctx: RenderCtx,
    render_fn: Optional[Renderer] = None,
) -> str:
    """Render a cached one-shot reasoning demonstration when available.

    The rationale loader retains the old ``task`` parameter for compatibility.
    Track C rationale caches should use the task key ``"summary"``.

    A missing rationale raises instead of falling back to the answer-only
    exemplar: a silent fallback would make a ``_trace`` variant identical to
    its plain one-shot counterpart and corrupt the trace ablation.
    """
    if ctx.shot is not Shot.ONE:
        return ""

    normalized_task = str(task or SUMMARY_TASK).strip().lower()

    shot = load_rationale(
        technique,
        normalized_task,
    )

    if shot is None:
        raise FileNotFoundError(
            "Trace variant requires a cached rationale, but none was found for "
            f"technique={technique!r}, task={normalized_task!r}. "
            "Run scripts.build_rationales_billsum_prompts before the benchmark."
        )

    renderer = render_fn or numbered

    return (
        "Example:\n"
        f"Reference model: {shot.source_model}\n"
        "Input bill:\n"
        f"{renderer(shot.exemplar_sentences)}\n\n"
        "Reasoning:\n"
        f"{shot.rationale.strip()}\n\n"
        "Correct output:\n"
        f'{{"selected_sentences": {list(shot.shown_indices)}}}\n\n'
        "---\n\n"
        "New bill:\n"
    )


def render(
    aspect: str,
    sentences: Sequence[str],
    ctx: RenderCtx,
    instructions: str,
    preamble: str = "",
    render_fn: Optional[Renderer] = None,
    header_override: Optional[str] = None,
    example_override: str = "",
) -> str:
    """Assemble a complete Track C prompt.

    ``aspect`` is retained for compatibility with Track A technique classes.
    For Track C it should be ``"summary"``.

    Parameters
    ----------
    aspect:
        Compatibility task label. Track C should pass ``"summary"``.
    sentences:
        Source-document sentences in their original order.
    ctx:
        Shot and cap configuration.
    instructions:
        Technique-specific instructions.
    preamble:
        Optional text placed before the exemplar and document.
    render_fn:
        Optional sentence renderer, such as a centrality-annotated renderer.
    example_override:
        Optional technique-specific demonstration block.
    """
    if not sentences:
        raise ValueError(
            "Cannot render an extractive-summarization prompt with no sentences."
        )

    renderer = render_fn or numbered

    parts = [
    (
        header_override.strip()
        if header_override is not None
        else header(aspect)
    ),
    "\n\n",
    ]
    if preamble and preamble.strip():
        parts.extend(
            [
                preamble.strip(),
                "\n\n",
            ]
        )

    exemplar_text = example_override or example_block(
        ctx,
        render_fn=render_fn,
    )

    if exemplar_text:
        parts.append(exemplar_text)

    parts.extend(
        [
            "Input bill:\n",
            renderer(sentences),
            "\n\n",
            "Instructions:\n",
            instructions.strip(),
            "\n\n",
            "Rules:\n",
            cap_bullet(ctx, aspect),
            SHARED_RULES,
            "\n\n",
            "Required return format:\n",
            RETURN_FORMAT,
        ]
    )

    return "".join(parts)