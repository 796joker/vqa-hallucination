---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
    background: #ffffff;
  }
  section.title {
    background: linear-gradient(135deg, #e0f7fa 0%, #e8f5e9 100%);
    color: #1a3a2a;
    text-align: center;
  }
  section.title h1 { font-size: 2.2em; margin-top: 1.5em; color: #1a3a2a; }
  section.title h2 { font-size: 1.1em; font-weight: normal; color: #37474f; }
  section.title p { font-size: 0.9em; color: #546e7a; }
  section.demo {
    background: linear-gradient(135deg, #ede7f6, #e8eaf6);
    color: #283593;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.demo h1 { font-size: 3em; color: #283593; }
  section.demo h2 { color: #5c6bc0; }
  section.demo p { color: #455a64; }
  section.ending {
    background: linear-gradient(135deg, #e0f7fa, #e8f5e9);
    color: #1a3a2a;
    text-align: center;
  }
  section.ending h1 { font-size: 2.4em; margin-top: 1.2em; color: #1a3a2a; }
  section.ending h3 { font-size: 1.3em; margin-top: 1em; color: #37474f; }
  section.ending p { font-size: 1em; color: #546e7a; }
  h1 { color: #1e3a8a; font-size: 1.6em; }
  h2 { color: #2563eb; font-size: 1.2em; }
  table { font-size: 0.75em; margin: 0 auto; }
  th { background: #1e3a8a; color: white; }
  strong { color: #dc2626; }
  .columns { display: flex; gap: 2em; }
  .col { flex: 1; }
  .highlight { background: #fef3c7; padding: 0.5em 1em; border-left: 4px solid #f59e0b; margin: 0.5em 0; }
  .card { background: #f1f5f9; border-radius: 8px; padding: 0.8em; margin: 0.3em 0; }
  .card-red { border-left: 4px solid #ef4444; }
  .card-yellow { border-left: 4px solid #f59e0b; }
  .card-green { border-left: 4px solid #22c55e; }
  .card-blue { border-left: 4px solid #3b82f6; }
  .metric-box { display: inline-block; text-align: center; padding: 0.5em 1.5em; margin: 0.3em; background: #eff6ff; border-radius: 8px; }
  .metric-box .num { font-size: 2em; font-weight: bold; color: #1e3a8a; }
  .metric-box .label { font-size: 0.8em; color: #64748b; }
---

<!-- _class: title -->

# 基于 SFT 与 DPO 的视觉语言模型幻觉缓解研究


李俊林、吴志文、吴承旭 | 2026年4月

---

# 问题背景与研究目标

<div class="columns">
<div class="col">

## VLM 幻觉问题

视觉语言模型会**生成图像中不存在的内容**：

- GPT-4V 幻觉率 **31.2%**
- LLaVA-1.5 幻觉率 **33.3%**
- 物体幻觉 > 属性幻觉 > 关系幻觉

<div class="highlight">
核心挑战：在降低幻觉的同时保持模型一般能力
</div>

</div>
<div class="col">

## 研究目标

1. SFT 数据**规模**如何影响幻觉？
2. DPO 能否**有效纠正**幻觉？
3. **判别能力** vs **生成质量**的关系？
4. 后训练是否**破坏一般能力**？

</div>
</div>

---

# 技术方案 Pipeline

```
Qwen3-VL-8B  ──>  SFT 阶段  ──>  DPO 阶段  ──>  四基准评估
  基座模型        LLaVA 150K     RLHF-V 5.7K     POPE / CHAIR
  LoRA 0.29%     5K/50K 对比     + RLAIF-V        MME / MMBench
```

| 组件 | 选型 | 说明 |
|:-----|:-----|:-----|
| 基座模型 | Qwen3-VL-**8B**-Instruct | 课程推荐 2B，我们选 8B 更强基线 |
| SFT 数据 | LLaVA-Instruct-150K | 与课程要求一致 |
| DPO 数据 | RLHF-V 5.7K + RLAIF-V | 人工标注 + AI 标注双数据源 |
| 评估基准 | POPE + CHAIR + MME + MMBench | 判别 + 生成 + 综合 四维互补 |

---

# 实验设计概览

<div style="text-align:center; margin: 0.5em 0;">
<div class="metric-box"><div class="num">25</div><div class="label">组模型</div></div>
<div class="metric-box"><div class="num">5</div><div class="label">维消融</div></div>
<div class="metric-box"><div class="num">4</div><div class="label">个基准</div></div>
<div class="metric-box"><div class="num">120</div><div class="label">GPU-hours</div></div>
</div>

**五维消融实验**：

| 维度 | 变量 | 关键问题 |
|:-----|:-----|:---------|
| LoRA Rank | r = 2 / 4 / **8** / 16 | 参数效率 vs 表达能力 |
| SFT 数据规模 | **5K** / 10K / 20K / 50K | 少即是多？ |
| DPO Beta | 0.05 / 0.1 / 0.5 / **1.0** | 约束强度 |
| 损失函数 | **Sigmoid** / Hinge / IPO | 鲁棒性对比 |
| 训练轮数 | **1ep** / 3ep / 5ep | 过拟合风险 |

补充实验：RLAIF-V (AI 标注) vs RLHF-V (人工标注) 同规模对比

---

<!-- _class: demo -->

# 现场演示

## Gradio Demo — 幻觉对比可视化

Base vs SFT vs DPO 输出对比 | 成功案例 + 失败案例

---

# 核心实验结果

| 模型 | POPE F1 ↑ | Yes% | CHAIR_i ↓ | CHAIR_s ↓ | MME |
|:-----|:---------:|:----:|:---------:|:---------:|:---:|
| Base (无训练) | 0.906 | 43.1% | 33.31% | 65.73% | 2008.0 |
| SFT 5K | **0.922** | 45.7% | 23.24% | 48.19% | 1899.0 |
| SFT 50K | 0.895 | 46.9% | 16.64% | 31.25% | — |
| DPO baseline (SFT50K+β0.1) | 0.780 | 28.7% | 18.88% | 39.31% | — |
| **True Optimal (SFT5K+β1.0)** | **0.889** | **41.3%** | **20.12%** | **38.10%** | **1990.5** |
| RLAIF-V 5.7K (AI 标注) | 0.918 | 44.8% | 19.49% | 35.48% | — |
| DPO-only (跳过 SFT) | 0.913 | 43.8% | 33.55% | 64.31% | — |

<div class="highlight">
<strong>True Optimal</strong>：CHAIR_i 从 33.31% → 20.12%（<strong>降低 39.6%</strong>），MME 保持 99.1%
</div>

---

# 发现一：少即是多

<div class="columns">
<div class="col">

![w:440](report/figures/figure_5_3_data_scale_curve.png)

</div>
<div class="col">

## SFT 阶段
- 5K (F1=**0.922**) > 50K (0.895)
- 过量数据引入 yes-bias

## DPO 阶段
- 5.7K (CHAIR_i=**19.49%**) > 20K (23.59%)
- 规模扩大 3.5 倍**反而增加幻觉**

**数据质量 > 数据数量，贯穿 SFT 和 DPO 的普适规律**

</div>
</div>

---

# 发现二：DPO-only 悖论 — "会判断" ≠ "会描述"

<div class="columns">

<div class="col">

![w:340](report/figures/paradox_1_pope.png)

</div>
<div class="col">

![w:340](report/figures/paradox_2_chair.png)

</div>
<div class="col">

![w:340](report/figures/paradox_3_mme.png)

</div>
</div>

- DPO-only 跳过 SFT：POPE **最佳** (0.900) 但 CHAIR **最差** (31.83%)
- 判别能力与生成质量可以完全脱节，**SFT 基础不可或缺**

---

# 发现三：AI 标注 vs 人工标注

从 RLAIF-V 83K 中随机采样 **5.7K**，与 RLHF-V 5.7K 对齐规模（单次采样）：

| 指标 | RLAIF-V (AI 标注) | RLHF-V (人工标注) | 差异 |
|:-----|:---:|:---:|:---:|
| POPE F1 ↑ | **0.918** | 0.889 | +3.3% |
| CHAIR_i ↓ | **19.49%** | 20.12% | -0.6pp |
| CHAIR_s ↓ | **35.48%** | 38.10% | -2.6pp |
| MMBench | 89.47% | — | 能力保持 |

- 在本次采样下，AI 标注表现与人工标注**基本持平，略有优势**
- 但仅为单次随机采样，结论存在方差，需多次采样验证
- 数据规模扩大反而有害：5.7K (CHAIR_i=19.49%) > 20K (23.59%)

---

# 工程经验与踩坑

<div class="columns">
<div class="col">

<div class="card card-red">

**Think 标签陷阱**
Qwen3-VL 输出 `<think>` 标签 → 91.6% 畸形输出，F1 误算为 0.098
→ 设置 `enable_thinking=False` + 后处理清洗

</div>

<div class="card card-yellow">

**Beta 调参**
β<0.1 训练崩溃（loss 飙至 11+）
→ 扩大搜索范围至 β=1.0，最终 **β=1.0** 表现最优

</div>

</div>
<div class="col">

<div class="card card-green">

**SFT 悖论**
直觉：更多 SFT 数据 = 更好
现实：50K 的 F1 (0.895) **低于** 5K (0.922)
→ 大量数据强化了 yes-bias (46.9%)

</div>

<div class="card card-blue">

**评估要多维**
DPO-only 在 POPE 排第 5，看似可接受
但 CHAIR 暴露严重幻觉 (31.83%)
→ 单一指标会**误导决策**

</div>

</div>
</div>

---

<!-- _class: ending -->

# 敬请老师同学们批评指正！

