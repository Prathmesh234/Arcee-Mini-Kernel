#!/bin/bash
# =============================================================================
# Serve the merged Trinity-Mini model with vLLM
#
# Usage:
#   1. Install deps:              uv sync
#   2. Download the model:        uv run python download_model_from_modal.py
#   3. Serve it:                  bash serve_trinity.sh
#
# Prerequisites:
#   - 2x NVIDIA GPUs with sufficient VRAM
#   - uv installed (curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - Model weights downloaded to ./trinity-mini/
# =============================================================================

set -e

MODEL_DIR="./trinity-mini"

# Check that model weights exist
if [ ! -d "$MODEL_DIR" ]; then
    echo "Model not found at $MODEL_DIR"
    echo "Run 'uv run python download_model_from_modal.py' first to download the weights."
    exit 1
fi

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found. Make sure NVIDIA drivers are installed."
    exit 1
fi
echo "GPUs detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install dependencies via uv
echo ""
echo "Installing dependencies..."
uv sync

# Serve the model with tensor parallelism across 2 GPUs
echo ""
echo "Starting vLLM server on port 8000..."
echo "Model: $MODEL_DIR (tensor-parallel=2)"
echo ""

uv run vllm serve "$MODEL_DIR" \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_r1 \
  --port 8000 \
  --tool-call-parser hermes
