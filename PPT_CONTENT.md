# PPT 逐页内容（复制到 PowerPoint/WPS 排版）

> 10 页，15 分钟演示
> 图片均在服务器 `report/figures/` 目录下，需先下载到本地
> 建议模板：深蓝/白色学术风格，标题 28-36pt，正文 18-22pt

---

## Slide 1: 标题页

**标题（36pt，加粗，居中）**：
基于 SFT 与 DPO 的视觉语言模型幻觉缓解研究

**副标题（22pt，居中）**：
大模型后训练技术 · 方向6 视觉问答幻觉

**作者信息（18pt，居中，页面底部）**：
李俊霖 | 2026年4月

---

## Slide 2: 问题背景

**标题（28pt）**：问题背景与研究目标

**左栏：VLM 幻觉问题**

VLM（视觉语言模型）会生成图像中不存在的内容：
- GPT-4V 幻觉率 31.2%
- LLaVA-1.5 幻觉率 33.3%
- 物体幻觉 > 属性幻觉 > 关系幻觉

**右栏：研究目标**

1. SFT 数据规模如何影响幻觉？
2. DPO 能否有效纠正幻觉？
3. 判别能力 vs 生成质量的关系？
4. 后训练是否破坏一般能力？

**底部一句话**：
目标：在降低幻觉的同时保持模型一般能力

---

## Slide 3: 技术方案

**标题（28pt）**：技术方案 Pipeline

**流程图（4 个方框，用箭头连接）**：

```
[Qwen3-VL-8B] → [SFT 阶段] → [DPO 阶段] → [四基准评估]
  基座模型        LLaVA 150K     RLHF-V 5.7K    POPE/CHAIR
  LoRA 0.29%     5K/50K 对比     + RLAIF-V       MME/MMBench
```

**下方表格（4列）**：

| 组件 | 选型 | 说明 |
|------|------|------|
| 基座模型 | Qwen3-VL-8B-Instruct | 课程推荐 2B，我们选 8B 更强基线 |
| SFT 数据 | LLaVA-Instruct-150K | 与课程要求一致 |
| DPO 数据 | RLHF-V (5.7K 人工) + RLAIF-V (AI) | 双数据源对比 |
| 评估基准 | POPE + CHAIR + MME + MMBench | 四维互补评估 |

---

## Slide 4: 实验设计

**标题（28pt）**：实验设计概览

**四个大数字（突出展示，横排）**：

| 25 组模型 | 5 维消融 | 4 个基准 | 120 GPU-hours |
|:---------:|:--------:|:--------:|:-------------:|

**消融维度列表**：

1. **LoRA Rank**：r=2 / 4 / 8 / 16（参数效率 vs 表达能力）
2. **SFT 数据规模**：5K / 10K / 20K / 50K（少即是多？）
3. **DPO Beta**：0.05 / 0.1 / 0.5 / 1.0（约束强度）
4. **损失函数**：Sigmoid / Hinge / IPO（鲁棒性对比）
5. **训练轮数**：1ep / 3ep / 5ep（过拟合风险）

**底部**：
补充实验：RLAIF-V (AI 标注) vs RLHF-V (人工标注) 对比

---

## Slide 5: Demo 分隔页

**大标题（36pt，居中）**：
现场演示

**副标题（22pt）**：
Gradio Demo — 幻觉对比可视化

**要点（18pt）**：
- Base vs SFT vs DPO 输出对比
- 成功案例 + 失败案例分析
- POPE / CHAIR 实时评测

> **讲解提示**：切换到 Gradio 界面或播放预录视频，约 5 分钟

---

## Slide 6: 核心结果

**标题（28pt）**：核心实验结果

**主表格**：

| 模型 | POPE F1 ↑ | Yes% | CHAIR_i ↓ | CHAIR_s ↓ | MME |
|------|-----------|------|-----------|-----------|-----|
| Base (无训练) | 0.906 | 43.1% | 33.31% | 65.73% | 2008.0 |
| SFT 5K | **0.922** | 45.7% | 23.24% | 48.19% | 1899.0 |
| SFT 50K | 0.895 | 46.9% | 16.64% | 31.25% | — |
| DPO baseline | 0.780 | 28.7% | 18.88% | 39.31% | — |
| **True Optimal** | **0.889** | 41.3% | **20.12%** | 38.10% | **1990.5** |
| RLAIF-V 5.7K | 0.918 | 44.8% | 19.49% | 35.48% | — |
| RLAIF-V only | 0.913 | 43.8% | 33.55% | 64.31% | — |

**底部总结（加粗）**：
True Optimal = SFT 5K + DPO β=1.0 1ep：CHAIR_i 从 33.31% 降至 20.12%（-39.6%），MME 保持 99.1%

**推荐图片**：`figure_4_3_chair_comparison.png`（放在表格右侧或下方）

---

## Slide 7: 关键发现 — "少即是多"

**标题（28pt）**：发现一：少即是多

**左侧放图**：`figure_5_3_data_scale_curve.png`
（SFT 数据规模 vs POPE F1 曲线，5K 是最优点）

**右侧要点**：

**SFT 阶段**：
- 5K (F1=0.922) > 10K > 20K > 50K (0.895)
- 过量数据引入 yes-bias（46.9%→理想 50%）

**DPO 阶段**：
- RLAIF-V 5.7K (CHAIR_i=19.49%) > 20K (23.59%)
- 规模扩大 3.5 倍反而增加幻觉

**底部结论（加粗，大字）**：
数据质量 > 数据数量，这是贯穿 SFT 和 DPO 的普适规律

---

## Slide 8: 关键发现 — DPO-only 悖论 & AI 标注

**标题（28pt）**：发现二 & 三

**上半部分：DPO-only 悖论**

放图：`figure_4_5_dpo_only_paradox.png`

要点：
- DPO-only（跳过 SFT）：POPE 优秀 (F1=0.900) 但 CHAIR 最差 (31.83%)
- "会判断"≠"会描述"：判别能力与生成质量可以脱节
- **结论：SFT 基础不可或缺**

**下半部分：AI 标注 vs 人工标注**

| 指标 | RLAIF-V (AI) | RLHF-V (人工) | 差异 |
|------|-------------|---------------|------|
| POPE F1 | 0.918 | 0.889 | **+3.3%** |
| CHAIR_i | 19.49% | 20.12% | **-0.6pp** |
| MMBench | 89.47% | — | 保持 |

**结论：相同规模 5.7K，AI 标注略优，支持 RLAIF 路线可行性**

---

## Slide 9: 踩坑经验

**标题（28pt）**：工程经验与踩坑

**四个要点（每个用不同颜色背景卡片）**：

**1. Think 标签陷阱** 🔴
- Qwen3-VL 生成 `<think>...</think>` 标签
- 导致 91.6% 输出畸形，POPE F1 被误算为 0.098
- 解决：设置 `enable_thinking=False` + 输出后处理清洗

**2. Beta 调参** 🟡
- DPO β<0.1 导致训练崩溃（loss 飙升至 11+）
- 文献默认 β=0.1 不是最优，扩展搜索到 β=1.0
- β=1.0 成为最优配置

**3. SFT 悖论** 🟢
- 直觉：更多 SFT 数据 = 更好
- 现实：50K 的 POPE F1 (0.895) 低于 5K (0.922)
- 原因：大量数据强化了 yes-bias (46.9%)

**4. 评估要多维** 🔵
- DPO-only 在 POPE 排第 5，看似可接受
- 但 CHAIR 暴露严重幻觉 (31.83%)
- 教训：单一指标会误导决策

---

## Slide 10: 总结与展望

**标题（36pt，白色，深蓝背景）**：总结与展望

**三大贡献（横排三个卡片）**：

| 发现 1 | 发现 2 | 发现 3 |
|--------|--------|--------|
| **少即是多** | **DPO-only 悖论** | **AI 标注可行** |
| SFT 5K > 50K | 判别好 ≠ 生成好 | RLAIF-V ≈ RLHF-V |
| DPO 5.7K > 20K | SFT 基础不可缺 | POPE +3.3% |
| 质量 > 数量 | 评估需多维验证 | 规模化标注可期 |

**核心数据（居中，18pt）**：
25 组模型 | 5 维消融 | 4 基准评估 | 120 GPU-hours
最优配置：SFT 5K + DPO β=1.0 → CHAIR_i 降低 39.6%，MME 保持 99.1%

**底部**：
代码仓库：github.com/796joker/vqa-hallucination

**最后一行（大号居中）**：
谢谢！欢迎提问

---

## 图片清单（需从服务器下载）

PPT 中建议使用的图片：

| 图片文件 | 用在哪页 | 内容 |
|---------|---------|------|
| `figure_5_3_data_scale_curve.png` | Slide 7 | SFT 数据规模曲线（核心发现图） |
| `figure_4_5_dpo_only_paradox.png` | Slide 8 | DPO-only 悖论双轴图 |
| `figure_4_3_chair_comparison.png` | Slide 6 | CHAIR 对比柱状图（可选） |
| `figure_4_1_pope_three_models.png` | Slide 6 | POPE 三模型对比（可选） |
| `figure_4_4_mme_capability.png` | Slide 6 | MME 能力雷达图（可选） |

**下载命令**：
```bash
scp research@115.190.215.236:/mnt/disk2/lijunlin/vqa-hallucination/report/figures/figure_5_3_data_scale_curve.png .
scp research@115.190.215.236:/mnt/disk2/lijunlin/vqa-hallucination/report/figures/figure_4_5_dpo_only_paradox.png .
scp research@115.190.215.236:/mnt/disk2/lijunlin/vqa-hallucination/report/figures/figure_4_3_chair_comparison.png .
scp research@115.190.215.236:/mnt/disk2/lijunlin/vqa-hallucination/report/figures/figure_4_1_pope_three_models.png .
scp research@115.190.215.236:/mnt/disk2/lijunlin/vqa-hallucination/report/figures/figure_4_4_mme_capability.png .
```

---

## 时间分配建议

| 页码 | 内容 | 时间 |
|------|------|------|
| 1 | 标题页 | 15s |
| 2 | 问题背景 | 2min |
| 3 | 技术方案 | 2min |
| 4 | 实验设计 | 1min |
| 5-Demo | 现场演示 | 5min |
| 6 | 核心结果 | 1.5min |
| 7 | 少即是多 | 1.5min |
| 8 | DPO悖论+AI标注 | 1.5min |
| 9 | 踩坑经验 | 1min |
| 10 | 总结Q&A | 2min |
| **合计** | | **~15min** |
