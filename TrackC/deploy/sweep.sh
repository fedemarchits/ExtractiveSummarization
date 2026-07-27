#!/bin/bash
# sweep.sh — run all 6 local models over both datasets on a fixed random 100,
# freeing each model's weights before the next so peak disk stays ~one model
# (~50 GB) instead of all six (~90 GB). Lets an 80 GB container run the full sweep.
#
# Each model is downloaded ONCE, used for both datasets, then its HF weights are
# purged (silver + dataset caches are kept, so they aren't regenerated).
#
# Env:
#   HF_TOKEN        required (gated Gemma/Qwen)
#   MAX_DOCS/SEED   default 100 / 42 (random 100, identical across models)
#   MODELS          model config (default configs/models.vast.yaml)
#   KEEP_WEIGHTS=1  skip purging (only if the container disk is large, >=130 GB)
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MAX_DOCS="${MAX_DOCS:-100}"
export SEED="${SEED:-42}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

MODELS_LIST="${MODELS_LIST:-qwen35_2b qwen35_4b qwen35_9b gemma4_e2b gemma4_e4b gemma4_12b}"
DATASETS="${DATASETS:-xsum cnndm}"

for M in $MODELS_LIST; do
    for DS in $DATASETS; do
        echo "=================== $M / $DS ==================="
        MODEL="$M" DATASET="$DS" bash deploy/run.sh
    done
    if [ "${KEEP_WEIGHTS:-0}" != "1" ]; then
        echo ">> purging weights for $M ($HF_HOME/hub/models--*)"
        rm -rf "$HF_HOME"/hub/models--* 2>/dev/null || true
    fi
done

echo "=== sweep complete ==="
echo "aggregate: python -m engine.report --results results --out results/summary.csv"
echo "analysis : python -m analysis.run_all"
