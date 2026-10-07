"""Generate rationale caches for the BillSum-oriented Track C prompt set.

Example:
    python -m scripts.build_rationales_billsum_prompts \
        --dataset billsum --model qwen35_397b
"""
from __future__ import annotations

from pathlib import Path


def main() -> None:
    from prompt_sets import activate_billsum_prompts

    activate_billsum_prompts()

    from scripts import build_rationales as base

    base.SYSTEM = (
        "You are an expert in extractive summarization of legislation. "
        "Provide concise reasoning and then a JSON answer."
    )

    def _bill_reasoning_instructions(technique: str) -> str:
        instructions = {
            "chain_of_thought": (
                "Reason step by step about the bill's main purpose, major provisions, "
                "requirements, affected parties, substantive effects, coverage, and redundancy."
            ),
            "self_ask": (
                "For each sentence, ask whether it contributes essential, non-redundant information "
                "about the bill's purpose, provisions, requirements, conditions, or effects."
            ),
            "scoring_based": (
                "Assign each sentence an internal summary-importance score from 1 to 5, considering "
                "legal/substantive importance, coverage, specificity, and redundancy."
            ),
            "salience_inference": (
                "Infer the bill's central purpose and salient provisions, then explain which sentences "
                "are necessary for a concise whole-bill summary."
            ),
        }
        if technique not in instructions:
            raise ValueError(f"No rationale-generation instructions for technique: {technique}")
        return instructions[technique]

    base._reasoning_instructions = _bill_reasoning_instructions

    original_experiment_path = base._experiment_path

    def _experiment_path(dataset: str) -> Path:
        if dataset == "billsum":
            return Path("configs/experiment_billsum_updated_prompts.yaml")
        return original_experiment_path(dataset)

    base._experiment_path = _experiment_path
    base.main()


if __name__ == "__main__":
    main()
