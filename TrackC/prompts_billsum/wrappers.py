"""Prompt text for orthogonal wrapper mechanisms.

Only dynamic_llm_capper needs a prompt (a second pruning pass). self_consistency
needs no new prompt — it re-samples an existing technique's prompt.
"""
from __future__ import annotations

from typing import Sequence

from .shared import RETURN_FORMAT, SHARED_RULES, header


# def dynamic_capper_prompt(
#     sentences: Sequence[str], aspect: str, selected_1b: Sequence[int]
# ) -> str:
#     """Ask the model to prune its own selection to the essential sentences,
#     choosing the count itself (no fixed K). Used instead of a fixed cap."""
#     sel = "\n".join(f"Sentence {i}: {sentences[i - 1]}" for i in selected_1b)
#     return (
#         f"{header(aspect)}\n\n"
#         f'You previously selected these candidate sentences for the "{aspect}" aspect:\n'
#         f"{sel}\n\n"
#         "Instructions:\n"
#         f'Keep ONLY the essential sentences that most directly express the "{aspect}" '
#         "aspect; drop redundant, weak, or off-aspect ones. Do not add new sentences and "
#         "do not pad to a fixed count — choose the number yourself from the content.\n\n"
#         f"Rules:\n{SHARED_RULES}\n\n"
#         f"Return format:\n{RETURN_FORMAT}"
#     )
def dynamic_capper_prompt(
    sentences: Sequence[str],
    task: str,
    selected_1b: Sequence[int],
) -> str:
    del task

    selected = "\n".join(
        f"Sentence {i}: {sentences[i - 1]}"
        for i in selected_1b
    )

    return (
        f"{header('summary')}\n\n"
        "You previously selected these candidate summary sentences:\n"
        f"{selected}\n\n"
        "Keep only the sentences essential to a concise and informative "
        "whole-document summary. Remove redundant, weak, background-only, "
        "or minor-detail sentences. Do not add new sentences.\n\n"
        f"Rules:\n{SHARED_RULES}\n\n"
        f"Return format:\n{RETURN_FORMAT}"
    )