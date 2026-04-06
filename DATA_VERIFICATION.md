# 实验数据验证报告

> 生成日期: 2026-04-06
> 验证方法: 从服务器 results/eval/*/pope_*.json 原始预测文件重新计算所有指标
> 验证脚本: 使用 eval_pope.py 相同的 think 标签清理正则 (`</?think>`) 重新解析

---

## 一、POPE 评估结果（从原始预测文件重新计算）

### 1.1 核心模型

| 模型 | 目录 | R_Acc | R_F1 | R_Yes | P_Acc | P_F1 | A_Acc | A_F1 |
|------|------|-------|------|-------|-------|------|-------|------|
| Base | eval/base | 0.9120 | 0.9055 | 0.4313 | 0.8887 | 0.8834 | 0.8700 | 0.8664 |
| SFT 50K (r=8) | eval/sft | 0.8987 | 0.8954 | 0.4687 | 0.8543 | 0.8562 | 0.8137 | 0.8230 |
| SFT+DPO (β=0.1, 3ep) | eval/dpo | 0.8193 | 0.7802 | 0.3220 | 0.8150 | 0.7761 | 0.8103 | 0.7718 |
| DPO-only | eval/dpo_only | 0.9077 | 0.9003 | 0.4257 | 0.8860 | 0.8797 | 0.8730 | 0.8678 |

### 1.2 SFT 数据规模消融

| 模型 | 目录 | R_F1 | P_F1 | A_F1 | R_Yes |
|------|------|------|------|------|-------|
| SFT 5K | eval/ablation_sft_data5k | 0.9220 | 0.8928 | 0.8625 | 0.4573 |
| SFT 10K | eval/ablation_sft_data10k | 0.9031 | 0.8789 | 0.8498 | 0.4457 |
| SFT 25K | eval/ablation_sft_data25k | 0.8926 | 0.8579 | 0.8278 | 0.4560 |
| SFT 50K | eval/sft | 0.8954 | 0.8562 | 0.8230 | 0.4687 |

### 1.3 SFT LoRA Rank 消融

| 模型 | 目录 | R_F1 | P_F1 | A_F1 | R_Yes |
|------|------|------|------|------|-------|
| r=4 | eval/ablation_sft_r4 | 0.8885 | 0.8480 | 0.8193 | 0.4567 |
| r=8 (baseline) | eval/sft | 0.8954 | 0.8562 | 0.8230 | 0.4687 |
| r=16 | eval/ablation_sft_r16 | 0.8755 | 0.8477 | 0.8119 | 0.4427 |
| r=32 | eval/ablation_sft_r32 | 0.8759 | 0.8515 | 0.8105 | 0.4560 |

### 1.4 DPO Beta 消融 (均基于 SFT 50K + sigmoid + 3 epochs)

| Beta | 目录 | R_F1 | P_F1 | A_F1 | R_Yes | 状态 |
|------|------|------|------|------|-------|------|
| 0.01 | eval/ablation_dpo_beta001 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 崩溃 |
| 0.05 | eval/ablation_dpo_beta005 | 0.0757 | 0.0757 | 0.0757 | 0.0197 | 崩溃 |
| 0.1 | eval/dpo | 0.7802 | 0.7761 | 0.7718 | 0.3220 | 正常 |
| 0.2 | eval/ablation_dpo_beta02 | 0.8281 | 0.8208 | 0.8125 | 0.3590 | 正常 |
| 0.5 | eval/ablation_dpo_beta05 | 0.8411 | 0.8334 | 0.8253 | 0.3703 | 正常 |
| 1.0 | eval/ablation_dpo_beta10 | 0.8459 | 0.8382 | 0.8292 | 0.3740 | 正常 |

### 1.5 DPO 损失函数消融 (均 β=0.1, 3 epochs)

| 损失函数 | 目录 | R_F1 | R_Yes | 状态 |
|---------|------|------|-------|------|
| Sigmoid | eval/dpo | 0.7802 | 0.3220 | 正常 |
| Hinge | eval/ablation_dpo_hinge | 0.7907 | 0.3297 | 正常 |
| IPO | eval/ablation_dpo_ipo | 0.0000 | 0.0000 | 崩溃 |

### 1.6 DPO Epoch 消融 & 组合优化

| 模型 | 目录 | R_F1 | P_F1 | A_F1 | R_Yes |
|------|------|------|------|------|-------|
| DPO 3ep (β=0.1) | eval/dpo | 0.7802 | 0.7761 | 0.7718 | 0.3220 |
| DPO 1ep (β=0.1) | eval/ablation_dpo_epoch1 | 0.8693 | 0.8565 | 0.8416 | 0.3950 |
| DPO optimal (SFT5K+β=1.0+3ep) | eval/ablation_dpo_optimal | 0.8837 | 0.8709 | 0.8606 | 0.4083 |
| DPO true optimal (SFT5K+β=1.0+1ep) | eval/dpo_true_optimal | 0.8894 | 0.8747 | 0.8619 | 0.4130 |

---

## 二、CHAIR 评估结果 (从 chair_results.json / chair_metrics.json)

| 模型 | 目录 | CHAIR_s | CHAIR_i | Recall | Captions | Objects |
|------|------|---------|---------|--------|----------|---------|
| Base | eval/base | 65.73% | 33.31% | 81.37% | 496 | 1696 |
| SFT 50K | eval/sft | 31.25% | 16.64% | 64.89% | 496 | 1082 |
| SFT+DPO (β=0.1) | eval/dpo | 39.31% | 18.88% | 77.27% | 496 | 1324 |
| DPO-only | eval/dpo_only | 61.69% | 31.83% | 79.35% | 496 | 1618 |
| True Optimal | eval/dpo_true_optimal | 38.10% | 20.12% | 74.24% | 496 | 1292 |
| SFT r=4 | eval/ablation_sft_r4 | 30.04% | 16.59% | 64.75% | 496 | 1079 |
| SFT r=16 | eval/ablation_sft_r16 | 31.05% | 17.07% | 64.32% | 496 | 1078 |
| SFT r=32 | eval/ablation_sft_r32 | 29.03% | 16.10% | 64.46% | 496 | 1068 |
| SFT 5K | eval/ablation_sft_data5k | 31.65% | 16.73% | 67.70% | 496 | 1130 |
| SFT 10K | eval/ablation_sft_data10k | 29.44% | 15.93% | 66.04% | 496 | 1092 |
| SFT 25K | eval/ablation_sft_data25k | 29.44% | 16.26% | 64.46% | 496 | 1070 |
| DPO β=0.2 | eval/ablation_dpo_beta02 | 39.52% | 20.03% | 77.55% | 496 | 1348 |
| DPO β=0.5 | eval/ablation_dpo_beta05 | 44.76% | 22.10% | 78.35% | 496 | 1398 |
| DPO β=1.0 | eval/ablation_dpo_beta10 | 43.15% | 22.04% | 78.13% | 496 | 1393 |
| DPO hinge | eval/ablation_dpo_hinge | 40.12% | 19.67% | 76.98% | 496 | 1332 |
| DPO epoch=1 | eval/ablation_dpo_epoch1 | 34.07% | 17.81% | 72.37% | 496 | 1224 |
| DPO optimal | eval/ablation_dpo_optimal | 37.70% | 21.11% | 69.64% | 496 | 1227 |

未评估 CHAIR: dpo_beta001, dpo_beta005, dpo_ipo (模型崩溃，无法生成有效描述)

---

## 三、MME 评估结果 (从 mme_metrics.json)

| 模型 | 目录 | Perception | Cognition | Total | CPR |
|------|------|-----------|-----------|-------|-----|
| Base | eval/base | 1801.5 | 206.5 | 2008.0 | 100.0% |
| SFT 5K | eval/ablation_sft_data5k | 1692.0 | 207.0 | 1899.0 | 94.6% |
| DPO-only | eval/dpo_only | 1763.5 | 201.0 | 1964.5 | 97.8% |
| True Optimal | eval/dpo_true_optimal | 1796.5 | 194.0 | 1990.5 | 99.1% |

仅评估 4 个关键模型。原因: 每个模型 MME 评估需 ~1h GPU，共享服务器资源有限，优先覆盖 pipeline 各阶段代表性模型。

---

## 四、训练日志摘要 (从 all_results.json)

| 模型 | 目录 | Epochs | Train Loss | Eval Loss | Runtime (s) |
|------|------|--------|-----------|-----------|-------------|
| SFT r=8 50K | sft/lora_r8 | 2.0 | 0.8883 | 0.8613 | 17,419 |
| SFT 5K | ablation/sft_data5k | 2.0 | 0.9381 | 0.9041 | 3,936 |
| SFT 10K | ablation/sft_data10k | 2.0 | 0.9103 | 0.8728 | 7,650 |
| SFT 25K | ablation/sft_data25k | 2.0 | 0.8888 | 0.8719 | 19,022 |
| SFT r=4 | ablation/sft_r4 | 2.0 | 0.8841 | 0.8615 | 49,732 |
| SFT r=16 | ablation/sft_lora_r16 | 2.0 | 0.8650 | 0.8541 | 40,371 |
| SFT r=32 | ablation/sft_lora_r32 | 2.0 | 0.8526 | 0.8516 | 39,105 |
| DPO β=0.1 3ep | dpo/lora_r8_beta01 | 3.0 | 0.3772 | - | 4,639 |
| DPO-only | ablation/dpo_only | 3.0 | 0.3903 | - | 8,449 |
| DPO β=0.01 | ablation/dpo_beta001 | 3.0 | 0.4678 | - | 11,752 |
| DPO β=0.05 | ablation/dpo_beta005 | 3.0 | 0.3992 | - | 9,158 |
| DPO β=0.2 | ablation/dpo_beta02 | 3.0 | 0.4223 | - | 7,171 |
| DPO β=0.5 | ablation/dpo_beta05 | 3.0 | 0.7415 | - | 7,750 |
| DPO β=1.0 | ablation/dpo_beta10 | 3.0 | 1.3913 | - | 7,929 |
| DPO hinge | ablation/dpo_loss_hinge | 3.0 | 0.3781 | - | 7,939 |
| DPO IPO | ablation/dpo_loss_ipo | 3.0 | 0.5078 | - | 7,954 |
| DPO epoch=1 | ablation/dpo_epoch1 | 1.0 | 0.6313 | - | 1,797 |
| DPO optimal | ablation/dpo_optimal | 3.0 | 1.2450 | - | 5,104 |
| DPO true optimal | ablation/dpo_true_optimal | 1.0 | 2.1427 | - | 1,107 |

---

## 五、已知数据差异说明

NEXT_STEPS.md 中记录的 Base 和 SFT 50K 模型 POPE 数值与服务器上当前 pope_*.json 文件计算结果有差异:

| 模型 | NEXT_STEPS F1 | 服务器验证 F1 | NEXT_STEPS Yes% | 服务器验证 Yes% |
|------|--------------|-------------|----------------|---------------|
| Base | 0.879 | 0.906 | 43.1% | 43.1% |
| SFT 50K | 0.855 | 0.895 | 52.1% | 46.9% |

可能原因: base 和 SFT 50K 的 pope 预测文件在后续实验（如 case study 生成）中被重新生成覆盖。其余 17 个模型的数据完全匹配。

**本文档中所有数值以服务器当前文件为准。**

---

## 六、文件清单

评估结果目录: /mnt/disk2/lijunlin/vqa-hallucination/results/eval/
- 20 个子目录，每个包含 pope_random.json, pope_popular.json, pope_adversarial.json
- 17 个子目录包含 chair_results.json 或 chair_metrics.json
- 4 个子目录包含 mme_metrics.json 和 mme_answers.json

训练结果目录:
- results/sft/lora_r8 (SFT baseline)
- results/dpo/lora_r8_beta01 (DPO baseline)
- results/ablation/* (16 个消融模型)
