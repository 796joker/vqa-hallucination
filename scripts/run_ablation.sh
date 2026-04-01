#!/bin/bash
# ============================================================
# Comprehensive Ablation Experiments
# Usage:
#   bash scripts/run_ablation.sh [group]
#
# Groups:
#   all        - Run everything (very long!)
#   pipeline   - G1: Training pipeline comparison
#   lora_rank  - G2a: LoRA rank ablation (r=4,16,32,64)
#   lora_target- G2b: LoRA target ablation (q_proj,v_proj vs all)
#   data_scale - G3: SFT data scaling (5K,10K,25K)
#   sft_lr     - G4a: SFT learning rate (5e-5, 2e-4)
#   sft_epoch  - G4b: SFT epochs (1, 3)
#   dpo_beta   - G5a: DPO beta sensitivity (0.01,0.05,0.2,0.5)
#   dpo_loss   - G5b: DPO loss function (hinge, ipo)
#   dpo_lr     - G5c: DPO learning rate (1e-6, 1e-5)
#   resolution - G6: Image resolution (128², 256²)
#   eval_only  - Only run POPE evaluation on existing checkpoints
# ============================================================

set -e
n# Ensure conda env binaries and CUDA libs are on PATH\nexport PATH=/mnt/disk3/conda/miniconda3/envs/zh/bin:\/mingw64/bin:/usr/bin:/c/Users/AS/bin:/d/msys64/ucrt64/Scripts:/c/Users/AS/AppData/Local/Programs/Microsoft VS Code:/d/Users/AS/AppData/Local/Programs/Python/Python312/Scripts:/d/Users/AS/AppData/Local/Programs/Python/Python312:/d/Users/AS/AppData/Local/JetBrains/Toolbox/scripts:/d/msys64/ucrt64/bin:/c/Python313/Scripts:/c/Python313:/d/Program Files/NVDIA GPU Computing Toolkit/CUDA/v11.8/bin:/d/Program Files/NVDIA GPU Computing Toolkit/CUDA/v11.8/libnvvp:/e/gurobi1101/win64/bin:/f/SIMULIA/Commands:/c/Program Files/Microsoft MPI/Bin:/f/VM17/bin:/c/Program Files/OpenSSH:/c/Windows/system32:/c/Windows:/c/Windows/System32/Wbem:/c/Windows/System32/WindowsPowerShell/v1.0:/c/Windows/System32/OpenSSH:/c/Program Files (x86)/NVIDIA Corporation/PhysX/Common:/c/Program Files/NVIDIA Corporation/NVIDIA NvDLISR:/c/WINDOWS/system32:/c/WINDOWS:/c/WINDOWS/System32/Wbem:/c/WINDOWS/System32/WindowsPowerShell/v1.0:/c/WINDOWS/System32/OpenSSH:/e/Program Files/北航云盘:/c:/c/Pro:/c/Program Files/Azure Data Studio/bin:/c/Program Files/dotnet:/c/mingw64/bin:/c/WI:/f/Maven/apache-maven-3.8.8-bin/apache-maven-3.8.8/bin:/c/ProgramData/chocolatey/bin:/d/Tailscale:/c/Program Files/Docker/Docker/resources/bin:/c/Program Files/NVIDIA Corporation/Nsight Compute 2022.3.0:/c/Program Files/Tailscale:/d/三夏小学期/sonar-scanner/bin:/d/P:/d/Program Files/MongoDB/Server/7.0/bin:/c/WINDOWS/system32:/c/WINDOWS:/c/WINDOWS/System32/Wbem:/c/WINDOWS/System32/WindowsPowerShell/v1.0:/c/WINDOWS/System32/OpenSSH:/d/Program Files/Java/jdk-23/bin:/c/Program Files (x86)/PDFtk/bin:/d/Program Files (x86)/sbt/bin:/d/Joern/joern-cli:/d/SoftWare/Coursier:/d/SoftWare/Coursier/InstallDir:/d/SoftWare/scala3-3.3.5/bin:/d/Program Files/ghidra_11.2.1_PUBLIC:/d/Program Files/elasticsearch-8.16.0/bin:/c/Program Files/MySQL/MySQL Shell 8.0/bin:/d/Program Files/LLVM/bin:/d/Program Files/Git LFS:/c/Program Files (x86)/Pulse Secure/VC142.CRT/X64:/c/Program Files (x86)/Pulse Secure/VC142.CRT/X86:/c/Program Files (x86)/Common Files/Pulse Secure/TNC Client Plugin:/d/nvm:/d/nvm/nodejs:/d/Program Files (x86)/Tencent/微信web开发者工具/dll:/cmd:/d/Program Files/Docker/Docker/resources/bin:/d/Program Files/Bandizip:/d/Program Files/cursor/resources/app/bin:/d/Program Files/nodejs:/d/nvm/nodejs/node_global:/d/Users/AS/AppData/Local/Programs/Trae/bin:/d/Users/AS/AppData/Local/Programs/Python/Python312/Scripts:/d/Users/AS/AppData/Local/Programs/Python/Python312:/c/Users/AS/.local/bin:/c/Program Files/Java/jdk-21.0.5+11/bin:/c/Users/AS/AppData/Local/Microsoft/WindowsApps:/c/Program Files/Azure Data Studio/bin:/d/Program Files/JetBrains/IntelliJ IDEA 2024.3.4/bin:/c/Users/AS/AppData/Local/Programs/Microsoft VS Code/bin:/e/Program Files/JetBrains/PyCharm 2022.3.3/bin:/d/IntelliJ IDEA Community Edition 2018.3.6/bin:/d/CLion 2023.2.1/bin:/d/anaconda3:/d/anaconda3/Scripts:/d/anaconda3/Library/mingw-w64/bin:/d/anaconda3/Library/usr/bin:/d/anaconda3/Library/bin:/d/Program Files/JetBrains/DataGrip 2023.2.3/bin:/d/Jmeter/apache-jmeter-5.6.2/bin:/d/Users/AS/AppData/Local/Programs/mongosh:/c/Users/AS/AppData/Local/Coursier/data/bin:/d/SoftWare/Coursier/InstallDir:/d/nvm:/d/nvm/nodejs:/f/texlive/2025/bin/windows:/d/Users/AS/AppData/Local/Programs/Antigravity/bin:/c/Users/AS/AppData/Roaming/npm\nexport LD_LIBRARY_PATH=/mnt/disk3/conda/miniconda3/envs/zh/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${PROJECT_DIR}"

GROUP="${1:-all}"
export CUDA_VISIBLE_DEVICES=4,5

# Helper: train with DDP
train() {
    local config="$1"
    local name="$2"
    echo ""
    echo ">>> Training: ${name}"
    echo "    Config: ${config}"
    echo "    Start: $(date)"
    mkdir -p logs/ablation
    FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
        llamafactory-cli train "${config}" > "logs/ablation/${name}.log" 2>&1
    echo "    Done:  $(date)"
}

# Helper: evaluate a checkpoint on POPE
evaluate() {
    local adapter="$1"
    local eval_name="$2"
    echo ">>> Evaluating: ${eval_name}"
    CUDA_VISIBLE_DEVICES=4 python eval/generate_pope_answers.py \
        --model_path "${MODEL_PATH}" \
        --adapter_path "${adapter}" \
        --pope_dir data/pope_data \
        --output_dir "results/eval/${eval_name}" > "logs/ablation/eval_${eval_name}.log" 2>&1

    python eval/eval_pope.py \
        --input_dir "results/eval/${eval_name}" \
        --output_dir "results/eval/${eval_name}" >> "logs/ablation/eval_${eval_name}.log" 2>&1
    echo "    Eval results: results/eval/${eval_name}/"
}

# Helper: evaluate base model (no adapter)
evaluate_base() {
    echo ">>> Evaluating: base (no adapter)"
    CUDA_VISIBLE_DEVICES=4 python eval/generate_pope_answers.py \
        --model_path "${MODEL_PATH}" \
        --pope_dir data/pope_data \
        --output_dir "results/eval/base" > "logs/ablation/eval_base.log" 2>&1

    python eval/eval_pope.py \
        --input_dir "results/eval/base" \
        --output_dir "results/eval/base" >> "logs/ablation/eval_base.log" 2>&1
}

# ==================================================================
# G1: Training Pipeline Comparison
#     Base → SFT-only → DPO-only → SFT+DPO
# ==================================================================
run_pipeline() {
    echo "=========================================="
    echo "  G1: Training Pipeline Comparison"
    echo "=========================================="

    # DPO-only (without SFT stage)
    train configs/qwen3vl_dpo_only.yaml "dpo_only"

    # Evaluations
    evaluate_base
    evaluate "results/sft/lora_r8" "sft"
    evaluate "results/ablation/dpo_only" "dpo_only"
    evaluate "results/dpo/lora_r8_beta01" "sft_dpo"
}

# ==================================================================
# G2a: LoRA Rank Ablation (r=4, 8*, 16, 32, 64)
# ==================================================================
run_lora_rank() {
    echo "=========================================="
    echo "  G2a: LoRA Rank Ablation"
    echo "=========================================="

    train configs/qwen3vl_sft_lora_r4.yaml "sft_r4"
    train configs/qwen3vl_sft_lora_r16.yaml "sft_r16"
    train configs/qwen3vl_sft_lora_r32.yaml "sft_r32"
    train configs/qwen3vl_sft_lora_r64.yaml "sft_r64"

    # Evaluate all ranks (r8 is baseline, already evaluated in pipeline)
    evaluate "results/ablation/sft_r4" "ablation_r4"
    evaluate "results/sft/lora_r8" "ablation_r8"
    evaluate "results/ablation/sft_lora_r16" "ablation_r16"
    evaluate "results/ablation/sft_lora_r32" "ablation_r32"
    evaluate "results/ablation/sft_r64" "ablation_r64"
}

# ==================================================================
# G2b: LoRA Target Ablation (q_proj,v_proj vs all*)
# ==================================================================
run_lora_target() {
    echo "=========================================="
    echo "  G2b: LoRA Target Ablation"
    echo "=========================================="

    train configs/qwen3vl_sft_target_qv.yaml "sft_target_qv"

    evaluate "results/ablation/sft_target_qv" "ablation_target_qv"
    evaluate "results/sft/lora_r8" "ablation_target_all"
}

# ==================================================================
# G3: SFT Data Scaling (5K, 10K, 25K, 50K*)
# ==================================================================
run_data_scale() {
    echo "=========================================="
    echo "  G3: SFT Data Scaling"
    echo "=========================================="

    train configs/qwen3vl_sft_data5k.yaml "sft_data5k"
    train configs/qwen3vl_sft_data10k.yaml "sft_data10k"
    train configs/qwen3vl_sft_data25k.yaml "sft_data25k"

    evaluate "results/ablation/sft_data5k" "ablation_data5k"
    evaluate "results/ablation/sft_data10k" "ablation_data10k"
    evaluate "results/ablation/sft_data25k" "ablation_data25k"
    evaluate "results/sft/lora_r8" "ablation_data50k"
}

# ==================================================================
# G4a: SFT Learning Rate (5e-5, 1e-4*, 2e-4)
# ==================================================================
run_sft_lr() {
    echo "=========================================="
    echo "  G4a: SFT Learning Rate"
    echo "=========================================="

    train configs/qwen3vl_sft_lr5e5.yaml "sft_lr5e5"
    train configs/qwen3vl_sft_lr2e4.yaml "sft_lr2e4"

    evaluate "results/ablation/sft_lr5e5" "ablation_sft_lr5e5"
    evaluate "results/sft/lora_r8" "ablation_sft_lr1e4"
    evaluate "results/ablation/sft_lr2e4" "ablation_sft_lr2e4"
}

# ==================================================================
# G4b: SFT Epochs (1, 2*, 3)
# ==================================================================
run_sft_epoch() {
    echo "=========================================="
    echo "  G4b: SFT Epochs"
    echo "=========================================="

    train configs/qwen3vl_sft_epoch1.yaml "sft_epoch1"
    train configs/qwen3vl_sft_epoch3.yaml "sft_epoch3"

    evaluate "results/ablation/sft_epoch1" "ablation_sft_epoch1"
    evaluate "results/sft/lora_r8" "ablation_sft_epoch2"
    evaluate "results/ablation/sft_epoch3" "ablation_sft_epoch3"
}

# ==================================================================
# G5a: DPO Beta Sensitivity (0.01, 0.05, 0.1*, 0.2, 0.5)
# ==================================================================
run_dpo_beta() {
    echo "=========================================="
    echo "  G5a: DPO Beta Sensitivity"
    echo "=========================================="

    train configs/qwen3vl_dpo_beta001.yaml "dpo_beta001"
    train configs/qwen3vl_dpo_beta005.yaml "dpo_beta005"
    train configs/qwen3vl_dpo_beta02.yaml "dpo_beta02"
    train configs/qwen3vl_dpo_beta05.yaml "dpo_beta05"

    evaluate "results/ablation/dpo_beta001" "ablation_beta001"
    evaluate "results/ablation/dpo_beta005" "ablation_beta005"
    evaluate "results/dpo/lora_r8_beta01" "ablation_beta01"
    evaluate "results/ablation/dpo_beta02" "ablation_beta02"
    evaluate "results/ablation/dpo_beta05" "ablation_beta05"
}

# ==================================================================
# G5b: DPO Loss Function (sigmoid*, hinge, ipo)
# ==================================================================
run_dpo_loss() {
    echo "=========================================="
    echo "  G5b: DPO Loss Function"
    echo "=========================================="

    train configs/qwen3vl_dpo_loss_hinge.yaml "dpo_loss_hinge"
    train configs/qwen3vl_dpo_loss_ipo.yaml "dpo_loss_ipo"

    evaluate "results/dpo/lora_r8_beta01" "ablation_loss_sigmoid"
    evaluate "results/ablation/dpo_loss_hinge" "ablation_loss_hinge"
    evaluate "results/ablation/dpo_loss_ipo" "ablation_loss_ipo"
}

# ==================================================================
# G5c: DPO Learning Rate (1e-6, 5e-6*, 1e-5)
# ==================================================================
run_dpo_lr() {
    echo "=========================================="
    echo "  G5c: DPO Learning Rate"
    echo "=========================================="

    train configs/qwen3vl_dpo_lr1e6.yaml "dpo_lr1e6"
    train configs/qwen3vl_dpo_lr1e5.yaml "dpo_lr1e5"

    evaluate "results/ablation/dpo_lr1e6" "ablation_dpo_lr1e6"
    evaluate "results/dpo/lora_r8_beta01" "ablation_dpo_lr5e6"
    evaluate "results/ablation/dpo_lr1e5" "ablation_dpo_lr1e5"
}

# ==================================================================
# G6: Image Resolution (128²=16384, 256²=65536, 512²=262144*)
# ==================================================================
run_resolution() {
    echo "=========================================="
    echo "  G6: Image Resolution"
    echo "=========================================="

    train configs/qwen3vl_sft_lowres.yaml "sft_lowres"
    train configs/qwen3vl_sft_midres.yaml "sft_midres"

    evaluate "results/ablation/sft_lowres" "ablation_lowres"
    evaluate "results/ablation/sft_midres" "ablation_midres"
    evaluate "results/sft/lora_r8" "ablation_highres"
}

# ==================================================================
# Evaluate-only mode: run POPE on all existing checkpoints
# ==================================================================
run_eval_only() {
    echo "=========================================="
    echo "  Evaluate All Existing Checkpoints"
    echo "=========================================="

    # Main pipeline
    evaluate_base
    [ -d "results/sft/lora_r8" ] && evaluate "results/sft/lora_r8" "sft"
    [ -d "results/dpo/lora_r8_beta01" ] && evaluate "results/dpo/lora_r8_beta01" "sft_dpo"

    # Check ablation checkpoints and evaluate if they exist
    for dir in results/ablation/*/; do
        name=$(basename "$dir")
        [ -d "$dir" ] && evaluate "$dir" "ablation_${name}"
    done
}

# ==================================================================
# Dispatch
# ==================================================================
case "${GROUP}" in
    pipeline)    run_pipeline ;;
    lora_rank)   run_lora_rank ;;
    lora_target) run_lora_target ;;
    data_scale)  run_data_scale ;;
    sft_lr)      run_sft_lr ;;
    sft_epoch)   run_sft_epoch ;;
    dpo_beta)    run_dpo_beta ;;
    dpo_loss)    run_dpo_loss ;;
    dpo_lr)      run_dpo_lr ;;
    resolution)  run_resolution ;;
    eval_only)   run_eval_only ;;
    all)
        run_pipeline
        run_lora_rank
        run_lora_target
        run_data_scale
        run_sft_lr
        run_sft_epoch
        run_dpo_beta
        run_dpo_loss
        run_dpo_lr
        run_resolution
        ;;
    *)
        echo "Unknown group: ${GROUP}"
        echo "Available: all pipeline lora_rank lora_target data_scale sft_lr sft_epoch dpo_beta dpo_loss dpo_lr resolution eval_only"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "  Ablation group '${GROUP}' complete!"
echo "=========================================="
echo "Run analysis: python eval/eval_ablation.py"
