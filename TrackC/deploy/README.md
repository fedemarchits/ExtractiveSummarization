# deploy/ — Track C on a vast.ai A100

Runs the Track C grid (whole-document extractive summarization on **XSum** and
**CNN/DailyMail**) on a single rented A100. The whole `TrackC/` folder is mounted
at `/workspace`; results persist in `TrackC/results/` through the mount.

Two datasets, each needing two silver splits (4 files total):
`{xsum,cnndm}_silver.parquet` (test) and `{xsum,cnndm}_validation_silver.parquet`
(validation — required by one-shot variants and trace-cache building).

## Files
- `Dockerfile.gpu` — torch 2.5.1 / CUDA 12.1 base + `requirements.txt` + NLTK punkt
- `vast_setup.sh` — bare-box path: installs deps + **modern vllm**, no Docker
- `run.sh` — prepares silver, optional traces, **preflight**, then `run.py` (inside container)
- `run_docker.sh` — docker wrapper; mounts the project, forwards tokens/keys/knobs
- `../scripts/preflight.py` — fails fast if silver / trace-cache / secrets are missing

## Prerequisites
- `TrackC/.env` with `HF_TOKEN=hf_...` (gated Gemma/Qwen). Add `OPENAI_BASE_URL` +
  `OPENAI_API_KEY` only if you run the endpoint tiers (27B/31B) or build traces.
- An A100 (40 GB fits every local tier at bf16; the 12B is the largest).

## Path A — bare box (simplest on vast.ai)
```bash
cd TrackC
HF_TOKEN=hf_xxx bash deploy/vast_setup.sh          # deps + modern vllm + nltk

# validate the whole pipeline FAST on a subsample first:
MODEL=qwen35_9b DATASET=xsum MAX_DOCS=500 bash deploy/run.sh
```
`run.sh` auto-generates the silver splits if missing, runs preflight, then the
benchmark. Drop `MAX_DOCS` for the full ~11k-doc test set (days per model).

## Path B — Docker
```bash
cd TrackC
docker build -f deploy/Dockerfile.gpu -t trackc-gpu:latest .
MODEL=qwen35_9b DATASET=xsum MAX_DOCS=500 bash deploy/run_docker.sh
```

## The 6 local models (resume-safe; one at a time on one A100)
```bash
for M in qwen35_2b qwen35_4b qwen35_9b gemma4_e2b gemma4_e4b gemma4_12b; do
  MODEL=$M DATASET=xsum  bash deploy/run.sh
  MODEL=$M DATASET=cnndm bash deploy/run.sh
done
```
`--models configs/models.vast.yaml` (the default in `run.sh`) puts every local
tier on **vllm** — do not fall back to `configs/models.yaml` (HF, ~3-4 s/doc).

## Trace variants (`*_trace`)
The grid includes 4 `_trace` variants that need a prebuilt rationale cache from
the reference model (`qwen35_397b`, endpoint). Either build it once per dataset:
```bash
BUILD_RATIONALES=1 MODEL=qwen35_9b DATASET=xsum bash deploy/run.sh   # needs OPENAI_*
```
…or set each `*_trace` technique to `[]` in `configs/grid.yaml` to skip them.
Preflight blocks the run (with the exact command) if traces are enabled but the
cache is empty — so you won't discover this mid-run.

## Knobs (env for run.sh / run_docker.sh)
| var | meaning |
| --- | --- |
| `MODEL` | model alias (required) |
| `DATASET` | `xsum` or `cnndm` (required) |
| `MODELS` | model config (default `configs/models.vast.yaml`) |
| `MAX_DOCS` | subsample size for silver gen (unset = full ~11k) |
| `GENERATE_SILVER=1` | force regenerate silver |
| `BUILD_RATIONALES=1` | build the `_trace` cache first (needs `OPENAI_*`) |
| `SKIP_PREFLIGHT=1` | bypass the safety check (not recommended) |

## Aggregate + analysis (after runs)
```bash
python -m engine.report --results results --out results/summary.csv   # per-variant table
python -m analysis.run_all                                            # Tables 10-12, CIs, significance, position-bias
```

## Notes
- `results/<dataset>/<model>/<variant>.jsonl` — raw per-doc output, gitignored;
  `.meta.json` (latency) is tracked, matching Track A.
- Full XSum + CNN/DM test = ~11k docs each × ~50 variants → **days per model**.
  Subsample (`MAX_DOCS`) for iteration; the analysis suite (bootstrap CIs,
  Wilcoxon) stays valid on a seeded sample.
