# 实验状态总览

> 最后更新: 2026-04-06

## 一、已完成的工作（截至 2026-03-30）

### 1.1 训练完成的模型（共 20 个配置）

**SFT 模型（8 个）：**
| 模型 | 配置 | 输出目录 | 状态 |
|------|------|---------|------|
| SFT r=8 (50K, baseline) | qwen3vl_sft_lora.yaml | results/sft/lora_r8 | ✅ |
| SFT r=4 | qwen3vl_sft_lora_r4.yaml | results/ablation/sft_r4 | ✅ |
| SFT r=16 | qwen3vl_sft_lora_r16.yaml | results/ablation/sft_lora_r16 | ✅ |
| SFT r=32 | qwen3vl_sft_lora_r32.yaml | results/ablation/sft_lora_r32 | ✅ |
| SFT data=5K | qwen3vl_sft_data5k.yaml | results/ablation/sft_data5k | ✅ |
| SFT data=10K | qwen3vl_sft_data10k.yaml | results/ablation/sft_data10k | ✅ |
| SFT data=25K | qwen3vl_sft_data25k.yaml | results/ablation/sft_data25k | ✅ |
| SFT data=50K (=baseline) | - | - | ✅ |

**DPO 模型（12 个）：**
| 模型 | Beta | Loss | Epochs | SFT 基础 | 输出目录 | 状态 |
|------|------|------|--------|---------|---------|------|
| DPO baseline | 0.1 | sigmoid | 3 | SFT 50K r=8 | results/dpo/lora_r8_beta01 | ✅ |
| DPO beta=0.01 | 0.01 | sigmoid | 3 | SFT 50K | results/ablation/dpo_beta001 | ✅ |
| DPO beta=0.05 | 0.05 | sigmoid | 3 | SFT 50K | results/ablation/dpo_beta005 | ✅ |
| DPO beta=0.2 | 0.2 | sigmoid | 3 | SFT 50K | results/ablation/dpo_beta02 | ✅ |
| DPO beta=0.5 | 0.5 | sigmoid | 3 | SFT 50K | results/ablation/dpo_beta05 | ✅ |
| DPO beta=1.0 | 1.0 | sigmoid | 3 | SFT 50K | results/ablation/dpo_beta10 | ✅ |
| DPO hinge | 0.1 | hinge | 3 | SFT 50K | results/ablation/dpo_loss_hinge | ✅ |
| DPO IPO | 0.1 | IPO | 3 | SFT 50K | results/ablation/dpo_loss_ipo | ✅ |
| DPO epoch=1 | 0.1 | sigmoid | 1 | SFT 50K | results/ablation/dpo_epoch1 | ✅ |
| DPO-only | 0.1 | sigmoid | 3 | 无 (base) | results/ablation/dpo_only | ✅ |
| DPO optimal | 1.0 | sigmoid | 3 | SFT 5K | results/ablation/dpo_optimal | ✅ |
| DPO true optimal | 1.0 | sigmoid | 1 | SFT 5K | results/ablation/dpo_true_optimal | ✅ |

### 1.2 评估完成（POPE + CHAIR + MME）

**POPE（全部 20 模型）：** 3 splits × 3000 questions = 9000 questions/model

**CHAIR（14 模型）：** 500 COCO val2014 images/model

**MME（4 关键模型）：** base, sft5k, true_optimal, dpo_only

### 1.3 报告和图表

- 技术报告: 8 章 ~25000 字 (中英文双版本)
- 图表: 16 张 + 6 张案例研究
- Case study: 40 个案例 (10图×4问×3模型)

### 1.4 Git 仓库

- 远程: `git@github.com:796joker/vqa-hallucination.git`
- 236 服务器 working tree: **clean**（2978 文件已提交）
- 3 次 commit:
  1. `c959d52` Initial commit
  2. `80a4552` Add Chinese report, figures, and visualization scripts
  3. `0166e89` Add processed datasets and evaluation data

---

## 二、发现的严重问题（2026-04-06）

### 问题 1: DPO 训练数据不符合课程要求 🔴

| | 课程要求 | 我们实际使用 |
|--|---------|-------------|
| 数据集 | **RLAIF-V** (openbmb/RLAIF-V-Dataset) | **RLHF-V** (llamafactory/RLHF-V) |
| 数据量 | **83K** preference pairs | **5.7K** preference pairs |
| 来源 | OpenBMB (清华) | LLaMA-Factory 内置 |

课程网站明确指定: "RLAIF-V Dataset (83K visual preference pairs for DPO)"

### 问题 2: 评估基准不符合课程要求 🔴

| | 课程要求 | 我们实际使用 |
|--|---------|-------------|
| 综合评估 | **MMBench** | **MME** |

课程网站指定: "MMBench (multi-modal comprehensive evaluation)"
我们用的 MME 是不同的基准（虽然也是多模态能力评估）。

### 问题 3: 模型规格不同 🟡

| | 课程要求 | 我们实际使用 |
|--|---------|-------------|
| 模型 | Qwen3-VL-**2B**-Instruct | Qwen3-VL-**8B**-Instruct |

可解释为"升级版实验"，严重程度较低。

---

## 三、服务器与资源状态

### 236 服务器 (ssh research@115.190.215.236)
- 主机名: `iv-yeat9jnu9s5i3z55xk6w`
- 3 块盘 (disk1/disk2/disk3), disk3 已满
- **所有实验结果都在这里**
- GPU: 8× A100-80GB, 全部高负载 (64-75GB 占用)
- Conda: `/mnt/disk3/conda/miniconda3/envs/zh/bin`

### 37 服务器 (ssh research@115.190.234.37)
- 主机名: `iv-ye8yq3ljb45i3z4ubwcy`
- 2 块盘 (disk1/disk2)
- 只有基础 SFT+DPO 两个模型，ablation/eval 为空
- 有 Qwen3-VL-8B 模型: `/mnt/disk2/lijunlin/downloads/models/Qwen3-VL-8B-Instruct`
- GPU: 8× A100-80GB, 几乎全满
  - GPU 1: ~24GB 空闲 (勉强够 eval, 不够训练)
  - 其他 GPU: <10GB 空闲

### 本地 (D:\硕士阶段\研一下\大模型后训练\Course design\vqa-hallucination)
- **不是 git 仓库**（之前用 scp 手动拷贝，不完整）
- results/eval/ 和 results/ablation/ 为空
- 报告和图表文件完整

---

## 四、核心实验发现（保留供参考）

1. **SFT "less is more"**: 5K > 10K > 25K > 50K (POPE F1 单调递减)
2. **DPO beta trade-off**: β 越高 → POPE F1 越好但 CHAIR 越高; β=0.5-1.0 甜蜜区
3. **DPO-only 悖论**: POPE F1=0.900 (最佳) 但 CHAIR_i=31.83% (最差)
4. **1 epoch DPO > 3 epoch**: 验证文献推荐
5. **True optimal (SFT5K+β=1.0+1ep)**: POPE F1=0.889, CHAIR_i=20.12%, MME CPR=99.1%
6. **IPO/低 beta 崩溃**: β≤0.05 和 IPO 导致模型退化
7. **LoRA rank 影响极小**: r=4~32 差异 <2%
8. **Knowledge catastrophic forgetting**: SFT 损害知识任务 -7.03pp
