#!/bin/bash
# run.sh — Track C: run one model on one dataset, with every prerequisite
# checked/prepared first (silver data, optional rationale cache, preflight).
#
# Required env:
#   MODEL=<alias>        e.g. qwen35_9b   (must exist in $MODELS)
#   DATASET=xsum|cnndm|billsum
# Optional env:
#   MODELS=configs/models.vast.yaml   default; vllm local tiers (fast on A100)
#   MAX_DOCS=<n>                       subsample size for silver gen (unset = full ~11k)
#   GENERATE_SILVER=1                 (re)generate silver even if it already exists
#   BUILD_RATIONALES=1                build the _trace rationale cache (needs OPENAI_* + credit)
#   PROMPT_SET=original|billsum       BillSum only; default original
#   SKIP_PREFLIGHT=1                  bypass the safety check (not recommended)
#
# Results -> results/<dataset>/<model>/<variant>.jsonl  (resume-safe).
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"
# CNN/DM downloads fail via HF's xet/CAS backend ("Request failed after N retries");
# force the standard download path.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
# Bound slow/throttled endpoint calls (e.g. 397b traces) so they fail fast + retry
# instead of hanging for minutes.
export OPENROUTER_TIMEOUT="${OPENROUTER_TIMEOUT:-90}"
mkdir -p /tmp && chmod 1777 /tmp 2>/dev/null || true
export TMPDIR=/tmp TEMP=/tmp TMP=/tmp

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${MODEL:?set MODEL=<alias> (see configs/models.vast.yaml)}"
: "${DATASET:?set DATASET=xsum, cnndm, or billsum}"
MODELS="${MODELS:-configs/models.vast.yaml}"
PROMPT_SET="${PROMPT_SET:-original}"

case "$DATASET" in
  xsum)  SILVER="data/silver/xsum_silver.parquet";  VAL_SILVER="data/silver/xsum_validation_silver.parquet"; EXEMPLAR_SPLIT="validation";;
  cnndm) SILVER="data/silver/cnndm_silver.parquet"; VAL_SILVER="data/silver/cnndm_validation_silver.parquet"; EXEMPLAR_SPLIT="validation";;
  billsum) SILVER="data/silver/billsum_silver.parquet"; VAL_SILVER="data/silver/billsum_train_silver.parquet"; EXEMPLAR_SPLIT="train";;
  *) echo "DATASET must be xsum, cnndm, or billsum, got '$DATASET'"; exit 2;;
esac

if [ "$PROMPT_SET" = "billsum" ] && [ "$DATASET" != "billsum" ]; then
    echo "PROMPT_SET=billsum is only valid with DATASET=billsum"
    exit 2
fi

# Random subsample: with MAX_DOCS set, take a reproducible random sample (SEED,
# default 42) so all models see the SAME documents. Unset MAX_DOCS = full split.
SEED="${SEED:-42}"
MAXDOCS_ARG=""
if [ -n "$MAX_DOCS" ]; then
    MAXDOCS_ARG="--max-docs $MAX_DOCS --seed $SEED"
fi

echo "=============================="
echo "Track C run"
echo "model   : $MODEL"
echo "dataset : $DATASET"
echo "models  : $MODELS"
echo "prompts : $PROMPT_SET"
echo "max_docs: ${MAX_DOCS:-<full>}$([ -n "$MAX_DOCS" ] && echo " (random seed $SEED)")"
echo "endpoint: ${OPENAI_BASE_URL:-<unset>}"
echo "=============================="

# vllm at runtime if the image didn't bake it in.
python -c "import vllm" 2>/dev/null || pip install -q -U vllm

# --- silver data: generate if missing (or forced) ---
if [ "${GENERATE_SILVER:-0}" = "1" ] || [ ! -f "$SILVER" ]; then
    echo ">> generating silver: $DATASET test $MAXDOCS_ARG"
    python -u generate_silver.py --dataset "$DATASET" $MAXDOCS_ARG
fi
if [ "${GENERATE_SILVER:-0}" = "1" ] || [ ! -f "$VAL_SILVER" ]; then
    echo ">> generating silver: $DATASET $EXEMPLAR_SPLIT $MAXDOCS_ARG"
    python -u generate_silver.py --dataset "$DATASET" --split "$EXEMPLAR_SPLIT" $MAXDOCS_ARG
fi

# --- rationale cache for _trace variants (optional; needs OPENAI_* + credit) ---
if [ "${BUILD_RATIONALES:-0}" = "1" ]; then
    echo ">> building reasoning traces ($DATASET)"
    if [ "$PROMPT_SET" = "billsum" ]; then
        python -u -m scripts.build_rationales_billsum_prompts --dataset "$DATASET" --models "$MODELS"
    else
        python -u -m scripts.build_rationales --dataset "$DATASET" --models "$MODELS"
    fi
fi

# --- preflight: fail fast before loading a model / burning GPU time ---
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
    python scripts/preflight.py --dataset "$DATASET" --prompt-set "$PROMPT_SET" --model "$MODEL" --models "$MODELS"
fi

# --- run ---
if [ "$PROMPT_SET" = "billsum" ]; then
    python -u run_billsum_prompts.py --model "$MODEL" --dataset "$DATASET" --models "$MODELS"
else
    python -u run.py --model "$MODEL" --dataset "$DATASET" --models "$MODELS"
fi
