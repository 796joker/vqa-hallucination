# VQA Hallucination Mitigation via SFT + DPO

基于 Qwen3-VL-8B-Instruct 的视觉问答幻觉缓解研究，通过 SFT（监督微调）和 DPO（直接偏好优化）两阶段后训练，系统探索幻觉产生机制与缓解策略。

## 核心发现

| 发现 | 详情 |
|------|------|
| **"少即是多"** | 5K SFT数据优于50K，POPE F1提升6.7pp（0.922 vs 0.855） |
| **知识灾难性遗忘** | SFT导致知识任务退化-7.03pp，DPO恢复+2.65pp |
| **DPO-only悖论** | POPE最佳(0.900)但CHAIR最差(31.83%)，判别≠生成 |
| **True Optimal** | SFT 5K + DPO β=1.0 1ep，三维最优平衡 |

## 项目结构

```
├── configs/          # LLaMA-Factory训练配置（28个YAML）
├── eval/             # 评估脚本（POPE/CHAIR/MME）
├── scripts/          # 运行脚本（训练/评估/数据处理）
├── demo/             # Gradio演示应用
├── data/             # 数据集元信息和POPE/COCO标注
├── results/
│   ├── ablation/     # 20个消融实验的LoRA adapter权重
│   ├── sft/          # SFT基线模型adapter
│   ├── dpo/          # DPO基线模型adapter
│   ├── eval/         # 所有模型的评估结果（JSON）
│   ├── case_studies/ # 案例研究输出
│   └── figures/      # 热力图等可视化
├── EXPERIMENT_PLAN.md
└── NEXT_STEPS.md
```

## 基座模型（未包含在仓库中）

| 模型 | 参数量 | 来源 | 说明 |
|------|--------|------|------|
| **Qwen3-VL-8B-Instruct** | 8.0B | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | 基座VLM，需下载至 `downloads/models/Qwen3-VL-8B-Instruct` |

## 数据集（未包含在仓库中）

| 数据集 | 用途 | 来源 | 预处理 |
|--------|------|------|--------|
| **LLaVA-Instruct-150K** | SFT训练 | [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | 经分层采样为5K/10K/25K/50K子集，保存至 `data/sft_data/` |
| **RLHF-V** | DPO训练 | [HaoyeZhang/RLHF-V-Dataset](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset) | 提取5,733偏好对，转换为LLaMA-Factory格式，保存至 `data/dpo_data/` |
| **COCO val2014** | CHAIR评估图像 | [cocodataset.org](https://cocodataset.org) | 需下载约6GB图像至 `data/mme_raw/`，标注文件已包含 |
| **MME Benchmark** | 能力评估 | [BradyFU/Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) | 解压至 `data/mme/`，包含14个子任务图像+问题 |
| **POPE** | 判别式幻觉评估 | 随COCO生成 | 已包含在 `data/pope_data/`（9K问题JSON） |

### 数据预处理

```bash
# 1. SFT数据准备（分层采样）
python data/prepare_sft_data.py --input data/llava_instruct_150k.json --sizes 5000 10000 25000 50000

# 2. DPO数据准备
python data/prepare_dpo_data.py --input data/rlhf_v_dataset.json

# 3. POPE数据准备
python data/prepare_pope.py --coco_ann data/coco_val2014_chair_annots.json
```

## 训练复现

**环境要求**: Python 3.12, PyTorch 2.5.0, CUDA 12.4, LLaMA-Factory 0.9.1

```bash
# SFT训练（单卡A100-40GB，约30分钟）
llamafactory-cli train configs/qwen3vl_sft_data5k.yaml

# DPO训练（双卡A100-40GB，约30分钟）
llamafactory-cli train configs/qwen3vl_dpo_true_optimal.yaml
```

**True Optimal配置**: SFT 5K (r=8, 2ep) → DPO β=1.0 (1ep, sigmoid)

## 评估复现

```bash
# POPE评估（判别式幻觉）
python eval/generate_pope_answers.py --model results/ablation/dpo_true_optimal
python eval/eval_pope.py --input results/eval/dpo_true_optimal/pope_answers.json

# CHAIR评估（生成式幻觉）
python eval/generate_chair_captions.py --model results/ablation/dpo_true_optimal
python eval/eval_chair.py --input results/eval/dpo_true_optimal/chair_captions.json

# MME评估（能力保持）
python eval/generate_mme_answers.py --model results/ablation/dpo_true_optimal
python eval/eval_mme.py --input results/eval/dpo_true_optimal/mme_answers/
```

## 消融实验配置

| 维度 | 变量 | 配置数 |
|------|------|--------|
| LoRA秩 | r ∈ {4, 8, 16, 32} | 4 |
| SFT数据规模 | {5K, 10K, 25K, 50K} | 4 |
| DPO Beta | β ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0} | 6 |
| 损失函数 | {Sigmoid, Hinge, IPO} | 3 |
| 训练阶段 | {SFT-only, DPO-only, SFT+DPO} | 3 |

共20个模型配置，adapter权重均保存在 `results/ablation/`。

## Git LFS

本仓库使用Git LFS管理以下大文件（>100MB）：
- `results/ablation/sft_lora_r16/adapter_model.safetensors` (167MB)
- `results/ablation/sft_lora_r32/adapter_model.safetensors` (334MB)

克隆时请确保安装git-lfs：
```bash
git lfs install
git clone git@github.com:796joker/vqa-hallucination.git
```

## License

MIT
