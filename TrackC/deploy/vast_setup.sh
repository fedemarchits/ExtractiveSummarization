#!/bin/bash
# vast_setup.sh — provision a fresh rented A100 box (vast.ai) for Track C.
#
# Installs deps with a MODERN vllm (the requirements pin 0.6.6.post1 is for the
# CUDA-12.4 3090 cluster, which does NOT support gemma-4 / qwen-3.5), fetches the
# NLTK data silver generation needs, and prints the GPU.
#
# Usage:
#   HF_TOKEN=hf_xxx bash deploy/vast_setup.sh
# Then see deploy/README.md for the generate-silver -> run sequence.
set -e

# Absolute project root (TrackC/), regardless of where this is invoked from.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${HF_TOKEN:?set HF_TOKEN=hf_... first (gemma / qwen are gated)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# Install deps EXCEPT the cluster-only pins: vllm/torch here must be the modern
# ones; bitsandbytes isn't needed (bf16 on a big card). Everything else is shared.
grep -ivE '^(vllm|torch|torchvision|torchaudio|bitsandbytes)' "$ROOT/requirements.txt" > /tmp/req_vast.txt
pip install -r /tmp/req_vast.txt
pip install -U vllm

# NLTK data for silver sentence splitting (silver.py falls back to regex, but
# punkt gives correct segmentation).
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

python - <<'PY'
import torch, vllm
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | vllm {vllm.__version__}")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu: {p.name}  {p.total_memory/1e9:.0f}GB")
PY

echo
echo "ready. next steps:"
echo "  # 1) generate silver (subsample first to validate the pipeline fast):"
echo "  python generate_silver.py --dataset xsum --max-docs 500"
echo "  python generate_silver.py --dataset xsum --split validation --max-docs 500"
echo "  # 2) run one model on one dataset (uses the vllm config):"
echo "  MODEL=qwen35_9b DATASET=xsum bash deploy/run.sh"
