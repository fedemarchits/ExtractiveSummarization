#!/bin/bash
# run_docker.sh — run a Track C script inside the GPU container with the TrackC/
# folder mounted at /workspace. Remaining args are forwarded to that script.
#
# Env:
#   IMAGE_NAME       docker image (default: trackc-gpu:latest)
#   HF_TOKEN         HuggingFace token (gated models: Gemma, some Qwen)
#   OPENAI_BASE_URL  } endpoint backend (large tiers) + build_rationales (397B)
#   OPENAI_API_KEY   }
#   MODEL, DATASET, MODELS, MAX_DOCS, GENERATE_SILVER, BUILD_RATIONALES  -> forwarded to run.sh
set -e

IMAGE_NAME="${IMAGE_NAME:-trackc-gpu:latest}"
SCRIPT="${1:-deploy/run.sh}"
shift || true

# Mount the TrackC/ root (parent of this deploy/ dir), not the cwd.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load secrets once from TrackC/.env (HF_TOKEN, OPENAI_*), if present.
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading secrets from .env"
    set -a
    # shellcheck disable=SC1090
    . "$PROJECT_ROOT/.env"
    set +a
fi

# --- GPU selection -------------------------------------------------------
# Forward exactly the GPUs allocated. Override manually with GPUS=... .
HOST_GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
NGPU="$(awk -F',' '{print NF}' <<< "$HOST_GPUS")"
CONTAINER_GPUS="$(seq 0 $((NGPU - 1)) | paste -sd, -)"

echo "Docker image : $IMAGE_NAME"
echo "Project root : $PROJECT_ROOT"
echo "Script       : $SCRIPT   args: $*"
echo "HF_TOKEN     : $([ -n "$HF_TOKEN" ] && echo set || echo UNSET)"
echo "endpoint     : ${OPENAI_BASE_URL:-<unset>}"
echo "GPUs         : host [$HOST_GPUS] -> container [$CONTAINER_GPUS]"

docker run --rm \
    --gpus "\"device=${HOST_GPUS}\"" \
    --shm-size=16g \
    -v "$PROJECT_ROOT":/workspace \
    -e HF_TOKEN="$HF_TOKEN" \
    -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
    -e OPENAI_BASE_URL="$OPENAI_BASE_URL" \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e MODEL="$MODEL" \
    -e DATASET="$DATASET" \
    -e MODELS="$MODELS" \
    -e MAX_DOCS="$MAX_DOCS" \
    -e SEED="$SEED" \
    -e GENERATE_SILVER="${GENERATE_SILVER:-0}" \
    -e BUILD_RATIONALES="${BUILD_RATIONALES:-0}" \
    -e SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}" \
    -e CUDA_VISIBLE_DEVICES="$CONTAINER_GPUS" \
    -e HF_HOME=/workspace/.cache/huggingface \
    "$IMAGE_NAME" \
    bash "$SCRIPT" "$@"
