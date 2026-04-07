#!/bin/bash
# Launch Gradio demo
# Run from project root: bash scripts/run_demo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_DIR}"

echo "=== Launching Gradio Demo ==="
echo "Base model: ${MODEL_PATH}"
echo "Adapter: results/ablation/dpo_true_optimal (SFT 5K + DPO beta=1.0 1ep)"
echo ""

# Default: load both models in bf16 (~32GB total, needs GPU with >=40GB free)
CUDA_VISIBLE_DEVICES=4 python demo/app.py \
    --model_path "${MODEL_PATH}" \
    --adapter_path results/ablation/dpo_true_optimal \
    --port 7860 \
    --share

# If OOM, use 4-bit quantization (~16GB total):
# CUDA_VISIBLE_DEVICES=4 python demo/app.py \
#     --model_path "${MODEL_PATH}" \
#     --adapter_path results/ablation/dpo_true_optimal \
#     --use_4bit \
#     --port 7860 \
#     --share
