# 3. Methodology

## 3.1 Base Model: Qwen3-VL-8B-Instruct

We adopt **Qwen3-VL-8B-Instruct** as our base model, a state-of-the-art vision-language model with:
- **Parameters**: 7.62B (8.0B including vision encoder)
- **Architecture**: Qwen2 language model + ViT-based vision encoder
- **Context Length**: 32K tokens
- **Training**: Pre-trained on 1.5T multimodal tokens, instruction-tuned on diverse VQA tasks
- **License**: Apache 2.0 (fully open-source)

**Selection Rationale**:
1. Strong baseline performance (POPE Acc 87.1%, competitive with LLaVA-1.5)
2. Efficient size for ablation studies (8B enables rapid experimentation)
3. Excellent multilingual support (English + Chinese)
4. Well-documented API via HuggingFace Transformers

**Model Path**: `../downloads/models/Qwen3-VL-8B-Instruct`

---

## 3.2 Supervised Fine-Tuning (SFT) Stage

### 3.2.1 Training Data

**Dataset**: LLaVA-Instruct-150K (filtered to 50K for main experiments)
- **Source**: HuggingFace `liuhaotian/LLaVA-Instruct-150K`
- **Format**: (image, question, detailed_answer) triplets
- **Composition**:
  - 90.3% descriptive questions ("Describe this image in detail")
  - 9.7% yes/no questions ("Is there a dog in the image?")
- **Filtering**: Removed low-quality/repetitive samples, retained high-diversity examples
- **Data Scale Variants**: 5K / 10K / 25K / 50K for ablation study

**Data Path**: `data/llava_instruct_150k.json`

### 3.2.2 LoRA Configuration

We use **Low-Rank Adaptation (LoRA)** for parameter-efficient fine-tuning:

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| **LoRA Rank (r)** | 8 (baseline) | Balances capacity and efficiency |
| **LoRA Alpha (α)** | 16 | Standard scaling factor (α=2r) |
| **Target Modules** | All linear layers | Comprehensive adaptation |
| **Trainable Params** | 22M (0.29% of total) | Highly parameter-efficient |

**Ablation**: Tested r ∈ {4, 8, 16, 32} to study rank sensitivity

### 3.2.3 Training Hyperparameters

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| **Learning Rate** | 5e-5 | Standard for LoRA SFT |
| **Batch Size** | 16 | Per-device batch size |
| **Gradient Accumulation** | 2 | Effective batch = 32 |
| **Epochs** | 2 | Prevents overfitting |
| **Optimizer** | AdamW | β1=0.9, β2=0.999 |
| **Scheduler** | Cosine | Warmup ratio = 0.03 |
| **Mixed Precision** | BF16 | Faster training |
| **Max Length** | 2048 | Tokens per sample |

**Training Time**:
- 5K data: ~30 minutes (GPU: A100-40GB)
- 50K data: ~5 hours

**Config Files**: `configs/qwen3vl_sft_*.yaml` (8 variants)

---

## 3.3 Direct Preference Optimization (DPO) Stage

### 3.3.1 Preference Data

**Dataset**: RLHF-V
- **Source**: HuggingFace `HaoyeZhang/RLHF-V-Dataset`
- **Size**: 5,733 preference pairs (image, prompt, chosen, rejected)
- **Format**: Binary preferences annotated by humans
- **Composition**:
  - **90.3%** descriptive responses (detailed captions)
  - **9.7%** yes/no responses
  - Yes:No ratio in yes/no subset: 290:525 (1.8:1 **no-bias**)
- **Quality**: Human-verified, high-agreement annotations

**Data Path**: `data/rlhf_v_5733.json`

### 3.3.2 DPO Hyperparameters

| Hyperparameter | Value | Ablation Range |
|----------------|-------|----------------|
| **Beta (β)** | 0.1 (baseline) | {0.01, 0.05, 0.1, 0.2, 0.5, 1.0} |
| **Loss Function** | Sigmoid | {Sigmoid, Hinge, IPO} |
| **Epochs** | 3 (baseline) | {1, 3} |
| **Learning Rate** | 5e-6 | Lower than SFT |
| **Batch Size** | 8 | Smaller for stability |
| **Gradient Accumulation** | 4 | Effective batch = 32 |

**Training Time**:
- 1 epoch: ~30 minutes
- 3 epochs: ~90 minutes

**Beta (β) Interpretation**:
- **β → 0**: Maximum likelihood (ignores preferences)
- **β → ∞**: KL constraint dominates (stays close to reference)
- **Optimal β**: Balances preference learning and stability

### 3.3.3 Training Pipeline

**Sequential Training** (Base → SFT → DPO):
1. Load base Qwen3-VL-8B-Instruct
2. Apply LoRA SFT on LLaVA-150K
3. Save SFT adapter
4. Load SFT adapter + apply DPO on RLHF-V
5. Save final DPO adapter (contains merged SFT+DPO weights)

**DPO-only Variant** (Base → DPO, skip SFT):
- Load base model directly
- Apply DPO on RLHF-V
- Tests if SFT is necessary

**Config Files**: `configs/qwen3vl_dpo_*.yaml` (20 variants)

---

## 3.4 Ablation Study Design

We conduct systematic ablation across **5 orthogonal dimensions**:

### 3.4.1 LoRA Rank Ablation (4 configs)

**Variable**: LoRA rank r ∈ {4, 8, 16, 32}
**Fixed**: SFT 50K data, 2 epochs; DPO β=0.1, 3 epochs
**Goal**: Determine minimal rank for hallucination mitigation
**Trainable Parameters**:
- r=4: 11M params
- r=8: 22M params (baseline)
- r=16: 44M params
- r=32: 87M params

### 3.4.2 SFT Data Scale Ablation (4 configs)

**Variable**: Data size ∈ {5K, 10K, 25K, 50K}
**Fixed**: LoRA r=8, 2 epochs; DPO β=0.1, 3 epochs
**Goal**: Test "more data → less hallucination" hypothesis
**Training Time**:
- 5K: 0.5h → 50K: 5h (10× speedup)

### 3.4.3 DPO Beta Ablation (6 configs)

**Variable**: β ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0}
**Fixed**: SFT 50K, r=8; DPO 3 epochs, sigmoid loss
**Goal**: Find optimal KL constraint strength
**Hypothesis**: Higher β → stronger yes-bias correction

### 3.4.4 Loss Function Ablation (3 configs)

**Variable**: Loss ∈ {Sigmoid, Hinge, IPO}
**Fixed**: SFT 50K, r=8; DPO β=0.1, 3 epochs
**Goal**: Compare DPO variants for VLM hallucination
**Loss Definitions**:
- **Sigmoid**: L = -log σ(β(r_w - r_l))
- **Hinge**: L = max(0, 1 - β(r_w - r_l))
- **IPO**: L = (r_w - r_l - 1/2β)²

### 3.4.5 Training Stage Ablation (3 configs)

**Variants**:
1. **SFT-only**: Base → SFT (no DPO)
2. **DPO-only**: Base → DPO (no SFT)
3. **SFT+DPO**: Base → SFT → DPO (full pipeline)

**Goal**: Determine necessity of each stage

### 3.4.6 True Optimal Configuration

**Definition**: Combines best hyperparameters from all ablations
- **SFT**: 5K data (from data scale ablation)
- **DPO**: β=1.0, 1 epoch (from beta + epoch ablations)
- **LoRA**: r=8 (sufficient from rank ablation)
- **Hypothesis**: Achieves global optimum across all metrics

**Total Configurations**: 4 + 4 + 6 + 3 + 3 + 1 = **21 models**
(Reduced to 20 after deduplication)

---

## 3.5 Evaluation Protocol

### 3.5.1 POPE (Discriminative Hallucination)

**Benchmark**: Polling-based Object Probing Evaluation
- **Size**: 9,000 yes/no questions (3 splits × 3,000)
- **Splits**:
  - **Random**: Objects randomly sampled from COCO vocabulary
  - **Popular**: High-frequency objects (person, car, tree)
  - **Adversarial**: Co-occurring objects (chair + table, fork + knife)
- **Metrics**:
  - **Accuracy**: Overall correctness
  - **Precision**: Positive predictive value
  - **Recall**: Sensitivity
  - **F1**: Harmonic mean of precision and recall
  - **Yes-Ratio**: Fraction of "yes" predictions (bias indicator)

**Ground Truth**: COCO object annotations
**Evaluation Script**: `eval/generate_pope_answers.py` + `eval/eval_pope.py`

### 3.5.2 CHAIR (Generative Hallucination)

**Benchmark**: Caption Hallucination Assessment with Image Relevance
- **Size**: 500 images from COCO val2014
- **Task**: Generate detailed image captions
- **Metrics**:
  - **CHAIR_s**: Sentence-level hallucination rate (% captions with ≥1 hallucination)
  - **CHAIR_i**: Instance-level hallucination rate (% hallucinated objects)
  - **Recall**: Coverage of ground-truth objects
  - **Num Objects**: Total objects mentioned (verbosity indicator)

**Ground Truth**: COCO object categories (80 classes)
**Evaluation Script**: `eval/generate_chair_captions.py` + `eval/eval_chair.py`

### 3.5.3 MME (Capability Preservation)

**Benchmark**: Multimodal Evaluation
- **Size**: 2,374 yes/no questions, 1,187 images
- **Subtasks**: 14 tasks across 2 categories
  - **Perception** (10 tasks): existence, count, position, color, posters, celebrity, scene, landmark, artwork, OCR
  - **Cognition** (4 tasks): commonsense_reasoning, numerical_calculation, text_translation, code_reasoning
- **Scoring**:
  - Accuracy (per-question correctness)
  - Accuracy+ (both paired questions correct)
  - Score = (acc + acc+) × num_images (max 2800)
- **Metrics**:
  - Perception Score (max 2000)
  - Cognition Score (max 800)
  - Total Score (max 2800)

**Ground Truth**: Human-annotated yes/no labels
**Evaluation Script**: `eval/generate_mme_answers.py` + `eval/eval_mme.py`

---

## 3.6 Infrastructure

**Hardware**:
- **GPU**: 4× NVIDIA A100-40GB (shared server)
- **CPU**: 64-core AMD EPYC
- **RAM**: 512GB

**Software**:
- **Framework**: LLaMA-Factory 0.9.1
- **Training**: DeepSpeed ZeRO-2
- **Inference**: vLLM 0.6.0
- **Python**: 3.12.0, PyTorch 2.5.0, CUDA 12.4

**Training Framework**: LLaMA-Factory
- Unified SFT/DPO training pipeline
- LoRA support with automatic adapter merging
- YAML-based configuration (28 config files)

**Reproducibility**:
- All configs in `configs/` directory
- Random seed: 42 (fixed)
- Deterministic CUDA operations enabled

---

**Summary**:
- **Base**: Qwen3-VL-8B-Instruct
- **SFT**: 50K LLaVA + LoRA r=8
- **DPO**: 5.7K RLHF-V + β=0.1
- **Ablations**: 5 dimensions, 20 configs
- **Evaluation**: POPE + CHAIR + MME (3-dimensional)
- **Training Time**: ~100 GPU-hours total
