# VQA 幻觉缓解实验计划

## 项目概述

**目标**：基于 Qwen3-VL-8B-Instruct，通过 SFT + DPO 两阶段后训练流程减少视觉问答中的幻觉现象，并通过系统性消融实验深入分析各因素的影响。

**基座模型**：Qwen3-VL-8B-Instruct (8B 参数)
**训练框架**：LLaMA-Factory
**硬件**：2× A100-80GB (DDP)
**评估基准**：POPE (Polling-based Object Probing Evaluation)

---

## 已完成

- [x] 数据准备：SFT 数据 (LLaVA-Instruct-150K → 50K samples)
- [x] 数据准备：POPE 评估数据 (9000 questions, 3 splits)
- [x] SFT 训练：LoRA r=8, 2 epochs, train_loss=0.8883, 4h50m
- [x] DPO 训练：beta=0.1, sigmoid loss, 3 epochs, train_loss=0.3772, 1h17m
- [x] 所有消融配置文件创建完毕 (19 configs)
- [x] 评估和可视化脚本就绪
- [x] 新服务器环境部署

---

## 实验阶段

### Phase 1：核心评估（课程展示最低要求）⏱️ ~2h

> **目标**：产出 Base vs SFT vs SFT+DPO 的 POPE 对比结果

| 步骤 | 命令 | 预计耗时 |
|------|------|---------|
| 1.1 评估 Base 模型 | `CUDA_VISIBLE_DEVICES=X python eval/generate_pope_answers.py --model_path ... --pope_dir data/pope_data --output_dir results/eval/base` | ~40min |
| 1.2 评估 SFT 模型 | 同上 + `--adapter_path results/sft/lora_r8` → `results/eval/sft` | ~40min |
| 1.3 评估 SFT+DPO 模型 | 同上 + `--adapter_path results/dpo/lora_r8_beta01` → `results/eval/sft_dpo` | ~40min |
| 1.4 生成对比报告和图表 | `python eval/eval_compare.py` | <1min |
| 1.5 生成幻觉分析 | `python eval/analyze_hallucination.py` | <1min |

**产出物**：
- `results/eval/{base,sft,sft_dpo}/` — 原始评估结果
- `results/figures/pope_accuracy_comparison.{pdf,png}` — 核心对比图
- `results/figures/pope_overall_comparison.{pdf,png}` — 整体指标图
- `results/figures/pope_yes_ratio.{pdf,png}` — Yes-Ratio 对比图
- `results/figures/hallucination_rate_comparison.{pdf,png}` — 幻觉率对比
- `results/figures/top_hallucinated_objects.{pdf,png}` — 高频幻觉对象

**或者一键执行**：
```bash
bash scripts/run_eval_all.sh
```

---

### Phase 2：定性案例分析（展示加分项）⏱️ ~1h

> **目标**：产出直观的 Before/After 对比案例，用于 PPT 展示

**之前方案中缺失，现补充。** POPE 是纯定量的 yes/no 评估，缺少让人直观感受幻觉减少的案例。

| 步骤 | 说明 |
|------|------|
| 2.1 挑选典型图片 | 从 COCO val2014 中选 5-8 张容易产生幻觉的图片（含多物体、遮挡、小物体） |
| 2.2 设计开放式问题 | 如 "Describe all objects in this image"、"What is happening in this scene?" |
| 2.3 三模型推理对比 | Base / SFT / SFT+DPO 分别回答同一问题 |
| 2.4 标注幻觉内容 | 人工标注每个回答中的幻觉部分（红色高亮） |
| 2.5 整理为展示素材 | 图片+三模型回答+标注 → 用于 PPT |

**需要新建的脚本**：`eval/generate_case_study.py` — 批量生成案例对比

---

### Phase 3：DPO 消融实验（简历核心亮点）⏱️ ~10h 训练 + ~5h 评估

> **目标**：系统研究 DPO 超参数对幻觉缓解的影响
>
> **优先级最高**：所有 DPO 消融共享同一个 SFT checkpoint，无需重新 SFT，训练快（每个 ~1.5h）

#### G1: 训练流程对比
| 实验 | 配置 | 训练时间 |
|------|------|---------|
| DPO-only (无SFT) | `qwen3vl_dpo_only.yaml` | ~1.5h |
| *(Base/SFT/SFT+DPO 已有)* | — | — |

**研究问题**：SFT 预训练是否是 DPO 有效的前提条件？

#### G5a: DPO Beta 敏感度
| β 值 | 配置 | 预期行为 |
|------|------|---------|
| 0.01 | `qwen3vl_dpo_beta001.yaml` | 弱 KL 约束，偏好学习激进 |
| 0.05 | `qwen3vl_dpo_beta005.yaml` | 较弱约束 |
| **0.1** | *(baseline, 已训练)* | 标准设置 |
| 0.2 | `qwen3vl_dpo_beta02.yaml` | 较强约束 |
| 0.5 | `qwen3vl_dpo_beta05.yaml` | 强 KL 约束，保守更新 |

**研究问题**：KL 散度约束强度如何影响幻觉率？β 过小是否导致过拟合？

#### G5b: DPO 损失函数
| 损失函数 | 配置 | 理论背景 |
|---------|------|---------|
| **Sigmoid (DPO)** | *(baseline, 已训练)* | Rafailov et al. 2023, 标准 DPO |
| Hinge | `qwen3vl_dpo_loss_hinge.yaml` | SVM 风格 margin loss |
| IPO | `qwen3vl_dpo_loss_ipo.yaml` | Azar et al. 2023, identity mapping |

**研究问题**：不同偏好优化目标函数对幻觉缓解效果的差异？

#### G5c: DPO 学习率
| 学习率 | 配置 |
|--------|------|
| 1e-6 | `qwen3vl_dpo_lr1e6.yaml` |
| **5e-6** | *(baseline, 已训练)* |
| 1e-5 | `qwen3vl_dpo_lr1e5.yaml` |

**运行方式**：
```bash
bash scripts/run_ablation.sh pipeline    # G1
bash scripts/run_ablation.sh dpo_beta    # G5a
bash scripts/run_ablation.sh dpo_loss    # G5b
bash scripts/run_ablation.sh dpo_lr      # G5c
```

---

### Phase 4：SFT 消融实验（简历深度亮点）⏱️ ~30h 训练 + ~8h 评估

> **目标**：研究 SFT 阶段各因素对下游幻觉缓解的影响
>
> 每个 SFT 实验需要重新训练（~2-5h/个），耗时较长

#### G2a: LoRA Rank
| Rank | 可训练参数量 | 配置 | 训练时间 |
|------|------------|------|---------|
| 4 | ~11M | `qwen3vl_sft_lora_r4.yaml` | ~4.5h |
| **8** | ~22M *(baseline)* | — | — |
| 16 | ~44M | `qwen3vl_sft_lora_r16.yaml` | ~5h |
| 32 | ~87M | `qwen3vl_sft_lora_r32.yaml` | ~5.5h |
| 64 | ~175M | `qwen3vl_sft_lora_r64.yaml` | ~6h |

**研究问题**：LoRA 秩与幻觉率的关系？是否存在最优秩？更高秩是否反而引入过拟合？

#### G2b: LoRA Target
| Target | 配置 |
|--------|------|
| q_proj, v_proj | `qwen3vl_sft_target_qv.yaml` |
| **all** *(baseline)* | — |

**研究问题**：微调全部线性层 vs 仅注意力投影层的效果差异？

#### G3: 数据缩放定律
| 数据量 | 配置 | 训练时间 |
|--------|------|---------|
| 5K | `qwen3vl_sft_data5k.yaml` | ~30min |
| 10K | `qwen3vl_sft_data10k.yaml` | ~1h |
| 25K | `qwen3vl_sft_data25k.yaml` | ~2.5h |
| **50K** *(baseline)* | — | — |

**研究问题**：SFT 数据量与 POPE 性能的缩放关系？边际收益在何处递减？

#### G4a: SFT 学习率
| 学习率 | 配置 |
|--------|------|
| 5e-5 | `qwen3vl_sft_lr5e5.yaml` |
| **1e-4** *(baseline)* | — |
| 2e-4 | `qwen3vl_sft_lr2e4.yaml` |

#### G4b: SFT 训练轮次
| Epochs | 配置 |
|--------|------|
| 1 | `qwen3vl_sft_epoch1.yaml` |
| **2** *(baseline)* | — |
| 3 | `qwen3vl_sft_epoch3.yaml` |

**研究问题**：训练轮次增加是否导致过拟合并加剧幻觉？

#### G6: 图像分辨率
| 分辨率 | max_pixels | 配置 |
|--------|-----------|------|
| 128² | 16,384 | `qwen3vl_sft_lowres.yaml` |
| 256² | 65,536 | `qwen3vl_sft_midres.yaml` |
| **512²** | 262,144 *(baseline)* | — |

**研究问题**：视觉输入的分辨率是否影响模型产生幻觉的倾向？

**运行方式**：
```bash
bash scripts/run_ablation.sh lora_rank   # G2a
bash scripts/run_ablation.sh lora_target # G2b
bash scripts/run_ablation.sh data_scale  # G3
bash scripts/run_ablation.sh sft_lr      # G4a
bash scripts/run_ablation.sh sft_epoch   # G4b
bash scripts/run_ablation.sh resolution  # G6
```

---

### Phase 5：训练动态对比分析（方案补充）⏱️ ~30min

> **之前方案中缺失，现补充。** 不仅比较最终 POPE 指标，还应比较不同设置下的训练过程。

| 分析内容 | 数据来源 |
|---------|---------|
| SFT loss 曲线对比（不同 rank/data_scale） | `results/ablation/*/training_loss.png` + TensorBoard |
| DPO loss + reward accuracy 曲线对比 | `results/ablation/*/training_rewards_accuracies.png` |
| 训练效率分析：参数量 vs 训练时间 vs 性能 | `trainer_state.json` 中的 train_runtime |

**需要新建的脚本**：`eval/compare_training_curves.py` — 叠加绘制不同实验的训练曲线

---

### Phase 6：汇总分析与出图 ⏱️ ~1h

```bash
# 一键分析所有消融结果
python eval/eval_ablation.py --eval_root results/eval --output_dir results/figures/ablation

# 导出汇总表格（CSV + JSON）
python eval/eval_ablation.py --group summary
```

**产出物**：
- `results/figures/ablation/g1_pipeline_overall.pdf` — 训练流程对比
- `results/figures/ablation/g2a_lora_rank.pdf` — LoRA Rank 曲线
- `results/figures/ablation/g3_data_scale.pdf` — 数据缩放曲线
- `results/figures/ablation/g5a_dpo_beta.pdf` — Beta 敏感度曲线
- `results/figures/ablation/g5b_dpo_loss.pdf` — 损失函数对比
- `results/figures/ablation/ablation_summary.csv` — 全部结果汇总表
- *(更多图表见脚本输出)*

---

### Phase 7：Demo 演示 ⏱️ ~30min

```bash
# 准备示例图片（从 COCO val2014 选取）
# 启动 Gradio Demo
CUDA_VISIBLE_DEVICES=X python demo/app.py --model_path ../downloads/models/Qwen3-VL-8B-Instruct --share
```

**注意**：Demo 需要同时加载 Base 和 SFT+DPO 两个模型，约占 2×16GB 显存。

---

### Phase 8：报告与 PPT ⏱️ ~3-5h

| 文档 | 内容 |
|------|------|
| 技术报告 | 动机→方法→实验→结果→分析→结论 |
| PPT (15min) | 问题定义→方法→核心结果→消融分析→Demo→总结 |

**PPT 建议结构**：
1. 视觉幻觉问题介绍 + 示例 (2min)
2. 方法：SFT + DPO 流程图 (2min)
3. 核心结果：Base vs SFT vs SFT+DPO (3min)
4. 消融实验亮点（选 2-3 组最有趣的结果）(4min)
5. 案例展示：Before/After 对比 (2min)
6. Demo 演示（如果时间允许）(2min)

---

## 实验依赖关系

```
Phase 1 (核心评估)     ← 无依赖，数据下好即可开始
Phase 2 (案例分析)     ← 需要 Phase 1 完成（确认模型可用）
Phase 3 (DPO 消融)     ← 需要 SFT baseline (已有)
Phase 4 (SFT 消融)     ← 无依赖，但耗时最长
Phase 5 (训练动态)     ← 需要 Phase 3 + Phase 4 部分完成
Phase 6 (汇总出图)     ← 需要各 Phase 评估结果
Phase 7 (Demo)         ← 需要模型权重（已有）
Phase 8 (报告 PPT)     ← 需要 Phase 1 + Phase 6 的图表
```

---

## 推荐执行顺序

### 课程展示优先路线（最小化时间）
```
Phase 1 → Phase 2 → Phase 7 → Phase 8
```
约需 **1 天**。产出：核心对比结果 + 案例 + Demo + PPT。

### 简历完善路线（完整实验）
```
Phase 1 → Phase 3 (与 Phase 2 并行)
       → Phase 4 (逐组跑，长期后台任务)
       → Phase 5 → Phase 6 → Phase 8
```
约需 **1-2 周**（主要是训练时间，可挂后台）。

---

## 已知问题与待修复项

1. **本地 `configs/qwen3vl_sft_lora.yaml` 仍有 `deepspeed` 行和 `resume_from_checkpoint: true`**
   - 新服务器上需要 sed 修复，或上传前本地修改

2. **`eval_pope.py` 接口不匹配**
   - 消融脚本调用 `--input_dir` 但 eval_pope.py 只接受 `--result_file`
   - 需要给 eval_pope.py 增加 `--input_dir` 批量模式

3. **缺少案例分析脚本**
   - 需要新建 `eval/generate_case_study.py`

4. **缺少训练曲线对比脚本**
   - 需要新建 `eval/compare_training_curves.py`

5. **Demo 缺少示例图片**
   - 需要在 `demo/examples/` 放入 3-5 张示例图片

6. **新服务器 CUDA_VISIBLE_DEVICES 需要根据实际空闲 GPU 调整**
   - 脚本中硬编码了 4,5，新服务器可能不同

---

## 预期成果总览

| 类别 | 数量 | 用途 |
|------|------|------|
| POPE 指标对比图 | 6+ 张 | 报告 + PPT |
| 消融曲线图 | 10+ 张 | 报告 + PPT |
| 案例对比展示 | 5-8 组 | PPT 重点展示 |
| 汇总数据表 (CSV) | 1 份 | 报告附录 |
| Gradio Demo | 1 个 | 现场演示 |
| 训练动态对比图 | 3-5 张 | 报告 |
