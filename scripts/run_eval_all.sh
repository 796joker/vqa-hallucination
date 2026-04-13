#!/bin/bash
# Run POPE evaluation for all three models: Base, SFT, SFT+DPO
# Run from project root: bash scripts/run_eval_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_DIR}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "=== Evaluating Base Model ==="
python eval/generate_pope_answers.py \
    --model_path "${MODEL_PATH}" \
    --pope_dir data/pope_data \
    --output_dir results/eval/base

echo ""
echo "=== Evaluating SFT Model ==="
python eval/generate_pope_answers.py \
    --model_path "${MODEL_PATH}" \
    --adapter_path results/sft/lora_r8 \
    --pope_dir data/pope_data \
    --output_dir results/eval/sft

echo ""
echo "=== Evaluating SFT+DPO Model ==="
python eval/generate_pope_answers.py \
    --model_path "${MODEL_PATH}" \
    --adapter_path results/dpo/lora_r8_beta01 \
    --pope_dir data/pope_data \
    --output_dir results/eval/sft_dpo

echo ""
echo "=== Generating Comparison Report ==="
python eval/eval_compare.py \
    --base_dir results/eval/base \
    --sft_dir results/eval/sft \
    --dpo_dir results/eval/sft_dpo \
    --output_dir results/figures

echo ""
echo "=== Hallucination Analysis ==="
python eval/analyze_hallucination.py \
    --base_dir results/eval/base \
    --sft_dir results/eval/sft \
    --dpo_dir results/eval/sft_dpo \
    --output_dir results/figures

echo ""
echo "=== All Evaluations Complete ==="
echo "Results: results/eval/"
echo "Figures: results/figures/"
