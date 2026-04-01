#!/bin/bash
# Launch Gradio demo
# Run from project root: bash scripts/run_demo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_DIR}"

echo "=== Launching Gradio Demo ==="
echo "Base model: ${MODEL_PATH}"
echo "Adapter: results/dpo/lora_r8_beta01"
echo ""

# Default: load both models in bf16 (~32GB total)
CUDA_VISIBLE_DEVICES=4 python demo/app.py \
    --model_path "${MODEL_PATH}" \
    --adapter_path results/dpo/lora_r8_beta01 \
    --port 7860 \
    --share

# If OOM, use 4-bit quantization:
# CUDA_VISIBLE_DEVICES=4 python demo/app.py \
#     --model_path "${MODEL_PATH}" \
#     --adapter_path results/dpo/lora_r8_beta01 \
#     --use_4bit \
#     --port 7860 \
#     --share
