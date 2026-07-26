# Prompting

Clean rebuild of the LLM extractive-summarization prompt experiments for the
expanded plan: **every technique in all 4 variants** (zero/one-shot x
capped/uncapped) across **Qwen3.5 + Gemma 4**, four matched size tiers each.

## Layout

| path                  | what                                                                       |
| --------------------- | -------------------------------------------------------------------------- |
| `prompts/`            | the core: techniques as a registry; variants generated in `base.py`        |
| `prompts/techniques/` | one file per technique (`vanilla.py` is the template)                      |
| `engine/`             | thin runtime (data, backends, metrics, runner) — STUBS, port behind tests  |
| `configs/`            | `models.yaml`, `experiment.yaml`, `grid.yaml` — the grid is data, not code |
| `data/shots/`         | few-shot exemplars, drawn from TRAIN only                                  |
| `results/`            | per-run output `results/<model>/<variant_slug>.jsonl`                      |
| `tests/`              | trust is earned here                                                       |
| `run.py`              | CLI                                                                        |

## Principles

1. **Prompts are the product** → isolated, one per file, no duplicated variant plumbing.
2. **Grid is config** → add a model or toggle a cell by editing YAML.
3. **Engine is thin and tested** → port from the old pipeline one module at a time.

## Techniques

9 existing: vanilla, least_to_most, tool_augmented, simulated_tool_augmented,
scoring_based, self_ask, chain_of_thought, explanation_based, salience_inference.

3 new: `self_critique` (one-shot form = self_refinement),
`contrastive_joint` (one-shot form = joint_self_ask),
`negative_aware` (one-shot only).

Orthogonal wrappers (not variants): `self_consistency` (majority vote),
`dynamic_llm_capper` (LLM prunes its own list instead of a fixed cap).

## Status

- [x] prompt registry + variant expansion (`prompts/base.py`, `registry.py`)
- [x] all 12 techniques (9 existing + 3 new), 46 variants
- [x] one-shot examples: answer-only + cached real reasoning traces (`rationale.py`)
- [x] few-shot exemplar selector (`fewshot.py`, TRAIN-only)
- [x] engine ported: data, postprocess, metrics, backends, grid, runner, report
- [x] TRAIN-median K capping
- [x] orthogonal wrappers: `self_consistency`, `dynamic_llm_capper` (config-toggled)
- [x] HF model ids verified on the Hub (`configs/models.yaml`)
- [x] endpoint access for large tiers (env: OPENAI_BASE_URL / OPENAI_API_KEY)
- [x] large-tier partial run: `qwen35_27b` + `gemma4_31b`, 5 best techniques (OpenRouter, ~$10)
- [ ] large-tier full grid (remaining 11 techniques, needs more endpoint credit)
- [ ] server run (GPU) + full pytest with deps installed

## Results so far

Aggregated from `results/summary.csv` (one row per model × variant × aspect).
Metrics: per-sentence **F1 / P / R** on selected indices, **ROUGE-L** of the built
summary, **oracle gap** (ROUGE-L below the best achievable selection; lower is
better). Aspect means include the per-doc `union` selection alongside the three
aspects. **All 8 grid models now have results**; the two largest (`qwen35_27b`,
`gemma4_31b`) are endpoint-only and were run on a **partial grid — the 5 best
techniques only** (budget-limited endpoint run, see
[§ Large tier](#large-tier--partial-run-5-best-techniques)).

### Where each model was computed

| Model        | GPU                       | Backend         | Precision | Status        |
| ------------ | ------------------------- | --------------- | --------- | ------------- |
| `qwen35_2b`  | RTX 3090 (faretra)        | HF transformers | bf16      | ✅ done       |
| `qwen35_4b`  | RTX 3090 (faretra)        | HF transformers | bf16      | ✅ done       |
| `qwen35_9b`  | A100 40GB (vast.ai)       | **vLLM**        | bf16      | ✅ done       |
| `gemma4_e2b` | RTX 3090 (faretra)        | HF transformers | bf16      | ✅ done       |
| `gemma4_e4b` | A100 40GB (vast.ai)       | **vLLM**        | bf16      | ✅ done       |
| `gemma4_12b` | A100 40GB (vast.ai)       | **vLLM**        | bf16      | ✅ done       |
| `qwen35_27b` | — (endpoint / OpenRouter) | endpoint        | full      | ⚠️ partial (5 tech) |
| `gemma4_31b` | — (endpoint / OpenRouter) | endpoint        | full      | ⚠️ partial (5 tech) |

gemma-4 is too new for the CUDA-12 cluster's vLLM, so `9b / e4b / 12b` were run
on a rented A100 (CUDA-13 → modern vLLM). All six local models are bf16, so
quality is comparable; **latency is not** (vLLM batches, HF does not; the endpoint
latency is round-trip wall-time, not compute).

The two large models were served via **OpenRouter** (`qwen/qwen3.5-27b`,
`google/gemma-4-31b-it`) at full precision, on a ~$10 budget that covered only the
**5 best techniques** (18 of 54 variants each). See
[§ Large tier](#large-tier--partial-run-5-best-techniques).

### Prompt variants computed (per model)

Each technique × **shot** (`zero`/`one`) × **cap** (`capped`/`uncapped`) — the
4-cell grid, except `negative_aware` and the `_trace` ablations (one-shot only).
**54 variants per model** — for the **6 local models**. The **2 large endpoint
models ran a partial grid: the 5 best techniques only = 18 variants each**
(`self_ask`, `salience_inference`, `self_critique`, `contrastive_joint`,
`negative_aware`; no `_trace` ablations). The full 54-cell grid is preserved in
`configs/grid.full.yaml`; the active partial grid is `configs/grid.yaml`.

| Technique                | zero·cap | zero·unc | one·cap | one·unc |
| ------------------------ | :------: | :------: | :-----: | :-----: |
| vanilla                  |    ✅    |    ✅    |   ✅    |   ✅    |
| chain_of_thought         |    ✅    |    ✅    |   ✅    |   ✅    |
| least_to_most            |    ✅    |    ✅    |   ✅    |   ✅    |
| self_ask                 |    ✅    |    ✅    |   ✅    |   ✅    |
| explanation_based        |    ✅    |    ✅    |   ✅    |   ✅    |
| salience_inference       |    ✅    |    ✅    |   ✅    |   ✅    |
| scoring_based            |    ✅    |    ✅    |   ✅    |   ✅    |
| tool_augmented           |    ✅    |    ✅    |   ✅    |   ✅    |
| simulated_tool_augmented |    ✅    |    ✅    |   ✅    |   ✅    |
| self_critique            |    ✅    |    ✅    |   ✅    |   ✅    |
| contrastive_joint        |    ✅    |    ✅    |   ✅    |   ✅    |
| negative_aware           |    —     |    —     |   ✅    |   ✅    |
| chain_of_thought_trace   |    —     |    —     |   ✅    |   ✅    |
| salience_inference_trace |    —     |    —     |   ✅    |   ✅    |
| scoring_based_trace      |    —     |    —     |   ✅    |   ✅    |
| self_ask_trace           |    —     |    —     |   ✅    |   ✅    |

### Results by model

Mean over all 54 variants × 3 aspects. `latency` = mean s/doc (backend-dependent,
not a cross-model comparison).

| Model        |  F1   | Precision | Recall | ROUGE-L | Oracle gap | Latency (s) |
| ------------ | :---: | :-------: | :----: | :-----: | :--------: | :---------: |
| `qwen35_2b`  | 0.461 |   0.438   | 0.674  |  0.137  |   0.263    |    4.48     |
| `qwen35_4b`  | 0.609 |   0.626   | 0.661  |  0.176  |   0.225    |    3.34     |
| `qwen35_9b`  | 0.623 |   0.641   | 0.678  |  0.177  |   0.223    |    0.84     |
| `gemma4_e2b` | 0.567 |   0.626   | 0.570  |  0.185  |   0.216    |    3.51     |
| `gemma4_e4b` | 0.611 |   0.647   | 0.639  |  0.179  |   0.222    |    0.43     |
| `gemma4_12b` | 0.631 |   0.642   | 0.681  |  0.179  |   0.222    |    1.18     |
| `qwen35_27b` |  †    |     †     |   †    |    †    |     †      |      †      |
| `gemma4_31b` |  †    |     †     |   †    |    †    |     †      |      †      |

† Large models ran only the 5 best techniques, so a 54-variant mean isn't
comparable — see [§ Large tier](#large-tier--partial-run-5-best-techniques) for
their stats and a like-for-like 5-technique comparison across all 8 models.

### Comparison 1 — capped vs. uncapped (per model)

| Model        | F1 capped | F1 uncapped | ROUGE-L capped | ROUGE-L uncapped |
| ------------ | :-------: | :---------: | :------------: | :--------------: |
| `qwen35_2b`  |   0.486   |    0.437    |   **0.178**    |      0.097       |
| `qwen35_4b`  |   0.602   |    0.616    |   **0.191**    |      0.160       |
| `qwen35_9b`  |   0.620   |    0.627    |   **0.194**    |      0.161       |
| `gemma4_e2b` |   0.558   |    0.576    |   **0.192**    |      0.177       |
| `gemma4_e4b` |   0.603   |    0.619    |   **0.187**    |      0.170       |
| `gemma4_12b` |   0.623   |    0.638    |   **0.192**    |      0.166       |

F1 is roughly flat (uncapped slightly higher except the weak 2B), but **ROUGE-L
is consistently better capped** — a fixed sentence budget stops over-selection
from diluting the summary.

### Comparison 2 — zero-shot vs. one-shot (per model, F1)

| Model        | zero-shot | one-shot |   Δ    |
| ------------ | :-------: | :------: | :----: |
| `qwen35_2b`  |   0.483   |  0.446   | −0.037 |
| `qwen35_4b`  |   0.605   |  0.612   | +0.007 |
| `qwen35_9b`  |   0.615   |  0.629   | +0.014 |
| `gemma4_e2b` |   0.589   |  0.552   | −0.037 |
| `gemma4_e4b` |   0.589   |  0.626   | +0.037 |
| `gemma4_12b` |   0.625   |  0.635   | +0.010 |

The exemplar **hurts the smallest models** (`2b`, `e2b`) and **helps as size
grows** — capacity to use a worked example appears with scale.

### How the rationale (trace) exemplars were built

The `_trace` variants show a one-shot example whose reasoning was produced
**once** by a strong reference model — **Qwen3.5-397B** (`qwen35_397b`, via
endpoint) — not by the grid model being evaluated. Built by
`scripts/build_rationales.py`:

1. For each reasoning technique × aspect, draw candidate exemplars from the
   **TRAIN** split only (never test).
2. Ask the 397B to solve the exemplar **and reveal its step-by-step reasoning**,
   ending with a parseable `Selected indices: [...]` line.
3. **Verify against human gold**: score the model's picks (F1). Try up to 10
   exemplars — early-accept at F1 ≥ 0.7, otherwise keep the best if it clears
   0.4; below that, that cell falls back to an answer-only one-shot.
4. Cache the `(exemplar, reasoning trace, gold)` once; **every grid model reuses
   the same frozen trace**.

So a trace is a *verified, gold-aligned reasoning demonstration from a large
model*, shown identically to every smaller model — which isolates the effect of
"showing **how** to reason" from each model's own capability.

### Comparison 3 — one-shot: answer-only vs. rationale (trace)

For the four techniques with a `_trace` variant (`chain_of_thought`,
`salience_inference`, `scoring_based`, `self_ask`): one-shot exemplar as a plain
answer vs. the 397B reasoning trace.

| Model        | answer-only F1 | trace F1 |     Δ      |
| ------------ | :------------: | :------: | :--------: |
| `qwen35_2b`  |     0.450      |  0.450   |   −0.000   |
| `qwen35_4b`  |     0.608      |  0.617   |   +0.008   |
| `qwen35_9b`  |     0.629      |  0.630   |   +0.001   |
| `gemma4_e2b` |     0.542      |  0.571   | **+0.029** |
| `gemma4_e4b` |     0.630      |  0.638   |   +0.008   |
| `gemma4_12b` |     0.635      |  0.632   |   −0.004   |

Rationale exemplars help most models slightly (largest gain on `gemma4_e2b`), but
the effect is small and not universal — no gain on the weakest (`2b`) or the
largest (`12b`).

### Large tier — partial run (5 best techniques)

The large tier is endpoint-only (OpenRouter, full precision). A ~$10 credit budget
covered the **5 best techniques by mean F1** — `self_ask`, `salience_inference`,
`self_critique`, `contrastive_joint`, `negative_aware` — i.e. **18 of 54 variants
each** (no `_trace` ablations). Both models completed all 18 with zero errors
(100 test docs × 3 aspects per variant).

**Full stats for the two large models** (mean over their 18 variants; incl. `union`):

| Model        |  F1   | Precision | Recall | ROUGE-L | Oracle gap | Latency (s)\* |
| ------------ | :---: | :-------: | :----: | :-----: | :--------: | :-----------: |
| `qwen35_27b` | 0.633 |   0.645   | 0.688  |  0.178  |   0.222    |     7.88      |
| `gemma4_31b` | 0.641 |   0.655   | 0.684  |  0.181  |   0.220    |     9.96      |

\* Endpoint round-trip wall-time (queueing + network), **not** compute — not
comparable to the local vLLM/HF latencies above.

**Like-for-like scaling** — mean F1 on the **same 5 techniques**, all 8 models, so
the large tier is directly comparable to the smaller ones:

| Tier        | Qwen3.5 model |  F1   | Gemma4 model |  F1   |
| ----------- | ------------- | :---: | ------------ | :---: |
| small       | `qwen35_2b`   | 0.472 | `gemma4_e2b` | 0.571 |
| small·med   | `qwen35_4b`   | 0.612 | `gemma4_e4b` | 0.629 |
| medium      | `qwen35_9b`   | 0.625 | `gemma4_12b` | 0.633 |
| **large**   | `qwen35_27b`  | **0.633** | `gemma4_31b` | **0.641** |

**Scaling is monotonic in both families** — every size step gains F1, no
inversions. Gemma4 leads Qwen3.5 at every tier. Returns diminish sharply past the
medium tier: Qwen `9b→27b` is **+0.008**, Gemma `12b→31b` is **+0.008**, vs the
huge low-end jump (Qwen `2b→4b` = +0.140). The extra parameters and full precision
of the large tier buy very little on this extractive task.

The two earlier per-model patterns **hold at the large tier** (5-technique means):

- **Capped vs. uncapped** — F1 flat/slightly higher uncapped (`27b` 0.628→0.639,
  `31b` 0.631→0.650), but **ROUGE-L clearly better capped** (`27b` 0.192 vs 0.164,
  `31b` 0.193 vs 0.169). Same story as the local models.
- **Zero- vs. one-shot** — one-shot ≥ zero-shot (`27b` 0.630→0.636, `31b`
  0.630→0.649), consistent with "exemplars help as size grows."

### Still to compute

- **Large-tier full grid** — the remaining 11 techniques (incl. `_trace`
  ablations) for `qwen35_27b` / `gemma4_31b`; needs more endpoint credit. Restore
  `configs/grid.full.yaml` when funded.
- **Raw `.jsonl` for `gemma4_e2b`, `qwen35_2b`, `qwen35_4b`** — computed on the
  vast.ai box; only their aggregated `summary.csv` rows were synced back. Pull the
  raw per-doc files before the instance is destroyed if error-analysis is needed.

### Top 5 prompts (mean F1 over the 6 local models)

This ranking is what **selected the 5 techniques** run on the large tier (the
`_trace` winners collapse to their base technique, since no `_trace` was run on
the endpoint models): `self_ask`, `salience_inference`, `self_critique`,
`contrastive_joint`, `negative_aware`.

| Rank | Technique                  |  F1   | Precision | Recall | ROUGE-L |
| :--: | -------------------------- | :---: | :-------: | :----: | :-----: |
|  1   | `self_ask_trace`           | 0.596 |   0.622   | 0.657  |  0.176  |
|  2   | `negative_aware`           | 0.593 |   0.626   | 0.639  |  0.178  |
|  3   | `salience_inference_trace` | 0.591 |   0.636   | 0.625  |  0.181  |
|  4   | `salience_inference`       | 0.591 |   0.631   | 0.629  |  0.180  |
|  5   | `self_critique`            | 0.590 |   0.601   | 0.662  |  0.171  |

Winners are the **reasoning-trace** and **precision-oriented** techniques (note
the high precision, 0.62–0.64, they probably win by cutting false positives?).

For reference:: the baseline `vanilla` is last at **0.558** and `tool_augmented`
second-last at 0.571.
