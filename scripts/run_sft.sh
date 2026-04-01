#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_DIR}"

echo "=== Stage 1: SFT Training ==="

RESUME_ARG=""
if ls results/sft/lora_r8/checkpoint-* 1>/dev/null 2>&1; then
    echo "Found checkpoint, resuming training..."
    RESUME_ARG="--resume_from_checkpoint true"
fi

FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=4,5 \
    llamafactory-cli train configs/qwen3vl_sft_lora.yaml ${RESUME_ARG}

echo "=== SFT Training Complete ==="
