"""Run Track C with the BillSum-oriented prompt package.

This launcher leaves the original ``prompts/`` package untouched. It aliases
``prompts_billsum/`` only for this process, then reuses the normal Track C
runner. For BillSum it also selects a separate experiment config/results tree,
so outputs cannot collide with runs using the original prompts.

Example:
    python run_billsum_prompts.py --model qwen35_4b --dataset billsum
"""
from __future__ import annotations



def main() -> None:
    from prompt_sets import activate_billsum_prompts

    activate_billsum_prompts()

    import run as standard_run

    standard_run.EXPERIMENT_CONFIGS["billsum"] = (
        "configs/experiment_billsum_updated_prompts.yaml"
    )
    standard_run.main()


if __name__ == "__main__":
    main()
