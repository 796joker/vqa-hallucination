#!/bin/bash
# Stage 2: DPO Training (builds on SFT checkpoint)
# Run from project root: bash scripts/run_dpo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_DIR}"

# Verify SFT checkpoint exists
if [ ! -d "results/sft/lora_r8" ]; then
    echo "ERROR: SFT checkpoint not found at results/sft/lora_r8"
    echo "Please run scripts/run_sft.sh first."
    exit 1
fi

echo "=== Stage 2: DPO Training ==="
echo "Config: configs/qwen3vl_dpo_lora.yaml"
echo "Base adapter: results/sft/lora_r8"
echo "Output: results/dpo/lora_r8_beta01"
echo ""

FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=4,5 llamafactory-cli train configs/qwen3vl_dpo_lora.yaml

echo ""
echo "=== DPO Training Complete ==="
echo "Checkpoint saved to: results/dpo/lora_r8_beta01"
