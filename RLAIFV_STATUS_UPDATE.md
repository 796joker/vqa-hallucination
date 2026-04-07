# RLAIF-V 补充实验状态更新

> 更新时间: 2026-04-07 (最终更新)
> 状态: Phase 1 训练完成 ✅, Phase 2 POPE完成 ✅, CHAIR 100%完成 ✅, MMBench 100%完成 ✅

---

## 当前完成度

| 阶段 | 任务 | 状态 | 完成度 |
|------|------|------|-------|
| **Phase 0** | 数据准备 | ✅ 完成 | 100% |
| **Phase 1** | DPO 训练 (5个模型) | ✅ 完成 | 100% |
| **Phase 2** | POPE 评估 (5×3 splits) | ✅ 完成 | 100% |
| **Phase 2** | CHAIR 评估 (5个模型) | ✅ 完成 | 100% (5/5) |
| **Phase 3** | MMBench 脚本编写 | ✅ 完成 | 100% |
| **Phase 3** | MMBench 评估执行 (4个模型) | ✅ 完成 | 100% (4/4) |

---

## 实时进度 (2026-04-07 11:10)

### ✅ 已完成

1. **训练 (5个模型, 2026-04-06 18:32 - 2026-04-07 05:47)**
   - baseline: 10h19m, loss=0.707
   - epoch1: 2h54m, loss=0.817
   - only: 9h58m, loss=0.611
   - optimal: 3h42m, loss=4.330
   - optimal_5k: 51m, loss=5.076

2. **POPE 评估 (15个文件, 完成于 2026-04-07 08:11)**
   - 最佳: optimal_5k (F1=0.9184)
   - 全部超越 Base 模型

3. **CHAIR 评估 (5个模型完成)**
   - optimal_5k: CHAIR_i=19.49% ✅ (优于 RLHF-V 20.12%)
   - optimal: CHAIR_i=23.59% ✅
   - only: CHAIR_i=33.55% ✅
   - baseline: CHAIR_i=18.83% ✅
   - epoch1: CHAIR_i=16.32% ✅ (所有模型中最低)

### ✅ 已完成（全部）

1. **编写 MMBench 评估脚本** ✅ (2026-04-07 15:50)
   - `data/prepare_mmbench.py`: 从 parquet 提取图片和问题
   - `eval/generate_mmbench_answers.py`: 多选题推理 + CircularEval 支持
   - `eval/eval_mmbench.py`: 准确率计算（overall + 20 类别 + L2 子类别）
2. **执行 MMBench 评估** ✅ (2026-04-07 16:46, 4个模型)
   - Base: 89.72%
   - SFT 5K: 89.35%
   - RLAIF-V optimal_5k: 89.47%
   - RLAIF-V only: 89.65%
   - 数据: MMBench-EN dev split, 4329 题, 20 个能力类别

---

## 核心发现 (可直接用于课程报告)

### 🔑 主要发现 1: AI标注 vs 人工标注

**控制规模实验 (5.7K 数据)**:
- RLAIF-V (AI标注): POPE F1=0.918, CHAIR_i=19.49%
- RLHF-V (人工标注): POPE F1=0.889, CHAIR_i=20.12%
- **结论**: AI标注在相同规模下略优 (+3.3% POPE, -0.6pp CHAIR)

### 🔑 主要发现 2: 数据规模 ≠ 性能提升

**规模对比 (RLAIF-V)**:
- 5.7K: POPE F1=0.918, CHAIR_i=19.49%
- 20K (3.5x规模): POPE F1=0.909, CHAIR_i=23.59%
- **结论**: 规模增加反而降低性能，质量 > 数量

### 🔑 主要发现 3: SFT 预训练的必要性

**DPO-only 对比**:
- POPE 表现良好 (F1=0.913, 排名第2)
- CHAIR 幻觉严重 (33.55%, 接近 Base 33.31%)
- **结论**: 直接 DPO 危险，SFT 基础不可缺

---

## 文档产出

1. **RLAIFV_DATA_VERIFICATION.md** — 完整数据验证报告
2. **RLAIFV_STATUS_UPDATE.md** — 本状态更新文档
3. **更新 RLAIFV_MMBENCH_PLAN.md** — 标记完成状态

---

## Git 同步计划

### 待推送文件 (~30 MB)

- ✅ 15 个 POPE JSON (每个 ~700KB)
- ✅ 3 个 CHAIR JSON (329KB + 466KB + 962KB)
- ✅ 训练日志 (5个, ~3MB 总计)
- ✅ 评估日志 (15个 POPE + 3个 CHAIR)
- ✅ 新增文档 (RLAIFV_DATA_VERIFICATION.md, RLAIFV_STATUS_UPDATE.md)

### 预期操作

```bash
cd /mnt/disk2/lijunlin/vqa-hallucination
git status
git add results/eval/dpo_rlaifv_*/*.json
git add logs/*rlaifv*.log
git add RLAIFV_*.md
git add RLAIFV_MMBENCH_PLAN.md
git commit -m "feat: RLAIF-V 补充实验完整数据 (训练+POPE+CHAIR 60%)

- 5个DPO模型训练完成 (baseline/epoch1/only/optimal/optimal_5k)
- POPE评估完成 (15个文件): optimal_5k最佳 F1=0.918
- CHAIR评估60%完成 (3/5): optimal_5k CHAIR_i=19.49%
- 核心发现: AI标注略优于人工标注 (同等5.7K规模)
- 数据验证: RLAIFV_DATA_VERIFICATION.md

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
git push origin main
```


## 全部评估完成 (2026-04-07)

### RLAIF-V 补充实验完整结果

| 模型 | SFT基础 | DPO规模 | POPE F1 | CHAIR_i | CHAIR_s | Recall | MMBench |
|------|---------|---------|---------|---------|---------|--------|---------|
| optimal_5k | SFT 5K | 5.7K | **0.9184** | 19.49% | 35.48% | 70.14% | 89.47% |
| baseline | SFT 50K | 20K×3ep | 0.9063 | 18.83% | 35.69% | 69.14% | — |
| epoch1 | SFT 50K | 20K×1ep | 0.8955 | **16.32%** | **30.24%** | 67.12% | — |
| optimal | SFT 5K | 20K | 0.9086 | 23.59% | 44.76% | 75.04% | — |
| only | 无 | 20K×3ep | 0.9126 | 33.55% | 64.31% | 80.36% | 89.65% |

### MMBench 完整结果 (4个关键模型)

| 模型 | MMBench Acc | 最强类别 | 最弱类别 |
|------|-----------|---------|---------|
| Base | **89.72%** | image_scene 98.0% | future_prediction 70.8% |
| SFT 5K | 89.35% | ocr 98.1% | image_quality 67.3% |
| RLAIF-V optimal_5k | 89.47% | ocr 98.7% | image_quality 68.7% |
| RLAIF-V only | 89.65% | image_scene 98.0% | spatial_relationship 70.6% |

### 核心发现

1. **epoch1 CHAIR最优**: 1 epoch训练的CHAIR_i=16.32%是所有模型中最低的
2. **AI标注 vs 人工标注** (同5.7K): AI标注POPE F1更高(+3.3%), CHAIR持平
3. **数据规模悖论**: 5.7K > 20K，质量优先于数量
4. **SFT必要性**: DPO-only的CHAIR_i=33.55%，接近Base水平
