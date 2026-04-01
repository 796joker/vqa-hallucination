# VQA 幻觉缓解实验计划 v2

## 项目定位

**核心叙事**：SFT 在提升任务能力的同时引入了 yes-bias 和幻觉倾向，DPO 通过偏好学习有效纠正这一问题。这构成 Base → SFT（问题暴露）→ SFT+DPO（问题解决）的三幕剧结构。

**学术参考**：
- HA-DPO (arXiv 2311.16839): beta=0.1, LoRA r=64, 幻觉感知偏好数据
- RLHF-V (arXiv 2312.00849): 密集段级 DPO, beta=0.5, 1.4K 样本即有效
- LLaVA-RLHF (arXiv 2309.14525): PPO 方法, LoRA r=64
- LRV-Instruction (arXiv 2306.14565): 证明正面数据偏差导致幻觉

---

## 评估体系（双 benchmark）

### Benchmark 1: POPE（判别式）
- 9000 个 yes/no 问题，3 个难度级别（Random/Popular/Adversarial）
- 指标：Accuracy, Precision, Recall, F1, **Yes-Ratio**
- Yes-Ratio 是关键：>50% 说明模型存在 yes-bias（幻觉倾向）

### Benchmark 2: CHAIR（生成式）
- 500 张 COCO val2014 图片，模型生成详细描述
- 指标：CHAIR_s（含幻觉的描述比例）, CHAIR_i（幻觉物体比例）, Recall（覆盖率）
- 互补 POPE：POPE 测判别能力，CHAIR 测生成时的幻觉控制
- 参考值：LLaVA-1.5-7B CHAIR_s=44.6%, CHAIR_i=12.8%（长描述）

### 补充分析
- **幻觉类型分布**：手动标注 50-100 个案例，分为物体/属性/关系幻觉
- **高频幻觉物体分析**：统计最常被幻觉的物体类别
- **Case Study**：5-8 组 Base/SFT/DPO 回答对比，高亮幻觉内容

---

## 已完成基线

| 模型 | POPE Random | POPE Popular | POPE Adversarial | 平均 | Yes-Ratio |
|------|-------------|--------------|------------------|------|-----------|
| Base (Qwen3-VL-8B) | 91.2% | 88.9% | 87.0% | 89.0% | 0.43-0.47 |
| SFT (LoRA r=8) | 89.9% | 85.4% | 81.4% | 85.6% | 0.47-0.55 |
| SFT+DPO | 评估中... | | | | |

**关键发现**：SFT 后 accuracy 下降 3.4%，yes-ratio 从 0.43→0.55，验证了 SFT 加剧幻觉的文献结论。

---

## 消融实验设计（精简版，聚焦高价值）

### 原则
1. 每组消融只变一个变量，其他保持 baseline 不变
2. 参数范围必须有文献依据
3. 砍掉可预测结果的低价值实验

### Group A: 训练流程对比（核心故事线）

| 实验 | 说明 | 状态 |
|------|------|------|
| Base | 无训练 | ✅ 已评估 |
| SFT-only | LoRA r=8, 50K, 2 epochs | ✅ 已评估 |
| DPO-only (无 SFT) | 直接在 base 上 DPO | ✅ 已训练 |
| SFT+DPO | 完整流程 | 🔄 评估中 |

**研究问题**：SFT 预训练是否是 DPO 有效的前提？DPO-only 能否直接减少幻觉？

### Group B: LoRA Rank 消融（SFT 阶段）

| Rank | 可训练参数 | 文献依据 | 状态 |
|------|-----------|---------|------|
| 4 | ~11M | 低秩极端 | 🔄 训练中 |
| **8** | ~22M | LLaMA-Factory 默认 | ✅ baseline |
| 16 | ~44M | | 🔄 训练中 |
| 32 | ~87M | | 🔄 训练中 |

**调整说明**：
- ~~r=64~~：砍掉。HA-DPO 用 r=64 但他们是在弱模型（MiniGPT-4）上，且 alpha=16（alpha/rank 比仅 0.25）。我们的 Qwen3-VL-8B 已经很强（Base 89%），r=32 已足够探索高秩区间，r=64 显存开销大且边际价值低。
- ~~target q_proj,v_proj~~：砍掉。LLaMA-Factory 官方和所有近期 VLM 论文都用 "all"，这已是共识，结果可预测。

**研究问题**：LoRA 秩是否存在最优值？低秩是否因欠拟合加剧幻觉，高秩是否因过拟合加剧？

### Group C: SFT 数据规模

| 数据量 | 训练时间 | 状态 |
|--------|---------|------|
| 5K | ~30min | 待跑 |
| 10K | ~1h | 待跑 |
| 25K | ~2.5h | 待跑 |
| **50K** | ~5h | ✅ baseline |

**文献依据**：RLHF-V 证明偏好数据 1.4K-5.7K 即有效；LLaVA-RLHF 用 50K SFT 数据。数据缩放曲线是课程项目的标准加分项。

**研究问题**：SFT 数据量与幻觉的关系？是否存在边际收益递减？更多 SFT 数据是否反而加剧幻觉（因为更多正面偏差）？

### Group D: DPO Beta 敏感度

| β 值 | KL 约束强度 | 文献依据 | 状态 |
|------|-----------|---------|------|
| 0.01 | 极弱 | 超出文献范围，作为极端对照 | ✅ 训练完成 |
| 0.05 | 弱 | DPO 原始论文测试范围下限 | 🔄 训练中 |
| **0.1** | 标准 | HA-DPO, LLaMA-Factory 默认 | ✅ baseline |
| 0.2 | 较强 | | 待跑 |
| 0.5 | 强 | RLHF-V 使用（full FT 场景） | 待跑 |

**调整说明**：保留 0.01 作为极端对照（已跑完不浪费），但论文中应注明 0.01 低于文献范围。0.5 来自 RLHF-V 但他们是 full FT，保留作为上限探索。

**研究问题**：KL 散度约束强度如何影响幻觉率？β 过小是否导致偏好过拟合？β 过大是否约束过强无法有效学习？

### Group E: DPO 损失函数

| 损失函数 | 理论背景 | 文献依据 | 状态 |
|---------|---------|---------|------|
| **Sigmoid (DPO)** | Rafailov et al. 2023 | 标准 baseline | ✅ baseline |
| Hinge (RSO/SLiC) | SVM 风格 margin loss | RSO 2023 | 待跑 |
| IPO | 解决过拟合问题 | Azar et al. 2023 (arXiv 2310.12036) | 待跑 |

**研究问题**：理论上 IPO 对噪声偏好更鲁棒，RLHF-V 数据可能有标注噪声，IPO 是否优于 sigmoid？

### ~~已砍掉的实验~~

| 实验 | 砍掉理由 |
|------|---------|
| SFT lr=5e-5 / 2e-4 | lr 消融对幻觉缓解的洞察有限，1e-4 是标准值 |
| SFT epoch=1 / 3 | 结果可预测（epoch↑→overfit），不是项目重点 |
| SFT target q_proj,v_proj | "all" 已是共识 |
| SFT lowres (128²) | 128×128 图像不可辨认，无学术意义 |
| SFT midres (256²) | 价值有限，且分辨率消融不是核心问题 |
| SFT r=64 | 在强 base 上边际价值低，显存开销大 |
| DPO lr=1e-6 / 1e-5 | lr 消融优先级低 |

**节省**：~70-90h GPU 时间，聚焦于高价值消融。

---

## 实验总量

| 类别 | 实验数 | 预计 GPU 时间 |
|------|--------|-------------|
| 训练流程对比 (A) | 4 个模型 | 已完成 |
| LoRA Rank (B) | 4 个 (r=4/8/16/32) | ~15h |
| 数据规模 (C) | 4 个 (5K/10K/25K/50K) | ~4h |
| DPO Beta (D) | 5 个 (0.01-0.5) | ~8h |
| DPO Loss (E) | 3 个 (sigmoid/hinge/IPO) | ~5h |
| **POPE 评估** | ~16 个模型 × 3 splits | ~16h |
| **CHAIR 评估** | 4 个核心模型 | ~4h |
| **总计** | ~16 个模型 | ~52h |

---

## 产出清单

### 核心图表（报告/PPT 必需）
1. **POPE 三模型对比柱状图**（Base/SFT/SFT+DPO × 3 splits）
2. **Yes-Ratio 变化图**（三模型 × 3 splits，展示 yes-bias 的引入和消除）
3. **CHAIR 对比图**（CHAIR_s, CHAIR_i, Recall 三指标）
4. **LoRA Rank vs POPE Accuracy 曲线**
5. **Data Scale vs POPE Accuracy 曲线**（期望看到边际递减）
6. **DPO Beta 敏感度曲线**（POPE acc + yes-ratio 双轴）
7. **DPO Loss 函数对比柱状图**
8. **训练流程消融柱状图**（Base/SFT/DPO-only/SFT+DPO）
9. **训练 Loss 曲线叠加图**（不同 rank / 不同 beta）
10. **Case Study 图**：3-5 组定性对比

### 报告章节结构
1. Introduction：VLM 幻觉问题 + 动机
2. Related Work：POPE, CHAIR, HA-DPO, RLHF-V
3. Method：SFT + DPO 流程，LoRA 配置
4. Experiments
   - 4.1 主实验：Base → SFT → SFT+DPO（POPE + CHAIR）
   - 4.2 发现：SFT 加剧幻觉（yes-bias 分析）
   - 4.3 消融 A：训练流程对比
   - 4.4 消融 B：LoRA Rank
   - 4.5 消融 C：数据规模
   - 4.6 消融 D：DPO Beta
   - 4.7 消融 E：DPO Loss
   - 4.8 定性分析 + Case Study
5. Conclusion

---

## 执行优先级

```
Priority 1 (立即): 完成 SFT+DPO POPE 评估 → 核心三模型结果
Priority 2 (本周): CHAIR 评估脚本 + 4 核心模型 CHAIR 评估
Priority 3 (并行): 跑完 LoRA rank (r4/r16/r32) + DPO beta 消融
Priority 4 (并行): SFT data scale (5K/10K/25K) — 训练快
Priority 5: DPO loss (hinge/IPO) + DPO beta (0.2/0.5)
Priority 6: 所有消融模型 POPE 评估
Priority 7: Case study + 分析出图
Priority 8: 报告 + PPT
```
