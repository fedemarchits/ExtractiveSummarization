# Track C — Results

Whole-document extractive summarization on **XSum** and **CNN/DailyMail**, silver labels (heuristic `local_score`, K=6).

**Run:** 100 random documents per dataset (seed 42), identical across models. Prompt grid: 52 variants (12 techniques + ablations + trace, × shot × cap). Metrics: sentence-index Precision/Recall/F1 vs silver, ROUGE-L of the reconstructed summary vs abstractive reference, oracle-gap (ROUGE-L below the greedy oracle).

**Models (bf16, vLLM):** Qwen3.5 {2B, 4B, 9B}, Gemma4 {E2B, E4B}. Gemma4-12B not yet run.

Raw per-document outputs (`results/<dataset>/<model>/*.jsonl`) are gitignored; `results/summary.csv` is the aggregate source.

---
## Mean over all 52 variants

### All variants (mean) — XSUM

| Model | P | R | F1 | ROUGE-L | Oracle gap |
| --- | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.620 | 0.777 | 0.664 | 0.084 | 0.095 |
| Qwen3.5-4B | 0.650 | 0.670 | 0.628 | 0.092 | 0.087 |
| Qwen3.5-9B | 0.651 | 0.675 | 0.640 | 0.089 | 0.090 |
| Gemma4-E2B | 0.632 | 0.502 | 0.551 | 0.086 | 0.093 |
| Gemma4-E4B | 0.710 | 0.517 | 0.586 | 0.103 | 0.077 |

### All variants (mean) — CNNDM

| Model | P | R | F1 | ROUGE-L | Oracle gap |
| --- | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.386 | 0.747 | 0.451 | 0.171 | 0.229 |
| Qwen3.5-4B | 0.432 | 0.592 | 0.460 | 0.198 | 0.201 |
| Qwen3.5-9B | 0.415 | 0.602 | 0.456 | 0.196 | 0.204 |
| Gemma4-E2B | 0.441 | 0.354 | 0.383 | 0.190 | 0.210 |
| Gemma4-E4B | 0.499 | 0.416 | 0.440 | 0.225 | 0.175 |

## Mean over capped variants (fixed ~K sentence budget)

### Capped variants (mean) — XSUM

| Model | P | R | F1 | ROUGE-L | Oracle gap |
| --- | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.611 | 0.585 | 0.596 | 0.093 | 0.086 |
| Qwen3.5-4B | 0.656 | 0.570 | 0.598 | 0.096 | 0.083 |
| Qwen3.5-9B | 0.666 | 0.595 | 0.622 | 0.094 | 0.085 |
| Gemma4-E2B | 0.662 | 0.508 | 0.568 | 0.090 | 0.089 |
| Gemma4-E4B | 0.710 | 0.516 | 0.587 | 0.102 | 0.077 |

### Capped variants (mean) — CNNDM

| Model | P | R | F1 | ROUGE-L | Oracle gap |
| --- | :-: | :-: | :-: | :-: | :-: |
| Qwen3.5-2B | 0.513 | 0.512 | 0.512 | 0.226 | 0.174 |
| Qwen3.5-4B | 0.492 | 0.462 | 0.474 | 0.228 | 0.172 |
| Qwen3.5-9B | 0.488 | 0.477 | 0.482 | 0.228 | 0.172 |
| Gemma4-E2B | 0.441 | 0.339 | 0.378 | 0.189 | 0.211 |
| Gemma4-E4B | 0.507 | 0.402 | 0.441 | 0.227 | 0.173 |

## Best-F1 variant per model

### Best-F1 variant per model — XSUM

| Model | Best F1 | Technique | shot | cap |
| --- | :-: | --- | :-: | :-: |
| Qwen3.5-2B | 0.745 | simulated_tool_augmented | one_shot | uncapped |
| Qwen3.5-4B | 0.746 | self_ask | zero_shot | uncapped |
| Qwen3.5-9B | 0.709 | explanation_based | one_shot | uncapped |
| Gemma4-E2B | 0.683 | self_ask_trace | one_shot | uncapped |
| Gemma4-E4B | 0.671 | chain_of_thought | one_shot | capped |

### Best-F1 variant per model — CNNDM

| Model | Best F1 | Technique | shot | cap |
| --- | :-: | --- | :-: | :-: |
| Qwen3.5-2B | 0.542 | tool_no_roleplay | zero_shot | capped |
| Qwen3.5-4B | 0.504 | tool_no_roleplay | zero_shot | capped |
| Qwen3.5-9B | 0.509 | salience_inference | one_shot | capped |
| Gemma4-E2B | 0.483 | tool_no_roleplay | zero_shot | capped |
| Gemma4-E4B | 0.477 | tool_augmented | zero_shot | capped |

