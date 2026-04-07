# 修复计划：课程要求对齐

> 创建日期: 2026-04-06
> 目标: 修复 DPO 训练数据 + 评估基准不匹配问题

---

## 需要修复的问题

| # | 问题 | 当前 | 应改为 | 优先级 |
|---|------|------|--------|--------|
| 1 | DPO 训练数据 | RLHF-V (5.7K) | RLAIF-V (83K) | P0 |
| 2 | 综合评估基准 | MME | MMBench | P0 |
| 3 | 模型规格 | 8B | 2B (课程要求) | P1 (可商议) |

---

## 修复方案

### 方案 A: 最小修复（推荐，~20-30h GPU）

保持 8B 模型，用 RLAIF-V 重跑关键 DPO 实验，补 MMBench 评估。
报告中说明模型规格升级的理由。

### 方案 B: 完全对齐（~60-80h GPU）

换 2B 模型，全部重来。代价过大，不推荐。

---

## 方案 A 详细步骤

### Phase 1: 数据准备（本地，无需 GPU）

#### 1.1 下载 RLAIF-V 数据集
```bash
# 本地下载（服务器无法访问外网）
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('openbmb/RLAIF-V-Dataset')
ds.save_to_disk('./rlaifv_raw')
"
```

#### 1.2 转换为 LLaMA-Factory 格式
项目已有转换脚本: `data/prepare_dpo_data.py`
```bash
python data/prepare_dpo_data.py --max_samples 83000 --output dpo_data/rlaifv_dpo.json
```

#### 1.3 下载 MMBench 数据集
```bash
# 从 OpenCompass 或 HuggingFace 下载 MMBench
# https://github.com/open-compass/MMBench
# 或 HuggingFace: lmms-lab/MMBench
```

#### 1.4 SCP 上传到服务器
```bash
scp -r rlaifv_dpo.json research@SERVER:/mnt/disk2/lijunlin/vqa-hallucination/data/dpo_data/
scp -r mmbench_data/ research@SERVER:/mnt/disk2/lijunlin/vqa-hallucination/data/mmbench/
```

### Phase 2: DPO 重训练（需要 GPU，~15-20h）

需要重跑的 DPO 实验（用 RLAIF-V 替代 RLHF-V）：

| 优先级 | 实验 | 配置修改 | 预估时间 | 预估显存 |
|--------|------|---------|---------|---------|
| P0 | DPO baseline (β=0.1, 3ep) | dataset → rlaifv_dpo | ~3h (DDP 2卡) | ~24GB/卡 |
| P0 | DPO β=1.0, 1ep (true optimal) | dataset → rlaifv_dpo, SFT=5K | ~1h (DDP 2卡) | ~24GB/卡 |
| P0 | DPO-only (β=0.1, 3ep) | dataset → rlaifv_dpo, 无SFT | ~3h (DDP 2卡) | ~24GB/卡 |
| P1 | DPO β=0.5, 3ep | dataset → rlaifv_dpo | ~3h | ~24GB/卡 |
| P1 | DPO β=1.0, 3ep | dataset → rlaifv_dpo | ~3h | ~24GB/卡 |
| P2 | DPO epoch=1 (β=0.1) | dataset → rlaifv_dpo | ~1h | ~24GB/卡 |
| P2 | DPO hinge (β=0.1) | dataset → rlaifv_dpo | ~3h | ~24GB/卡 |

**注意**: RLAIF-V 有 83K pairs（RLHF-V 只有 5.7K），训练时间会显著增加。
可能需要调整 max_samples 或 steps 来控制训练时间。

**配置修改要点**:
- `dataset: rlaifv_dpo` (替代 `rlhf_v`)
- 其他超参数保持不变，确保消融实验可比性
- SFT 阶段不需要改动（LLaVA-Instruct-150K 符合课程要求）

### Phase 3: 评估（需要 GPU，~5-8h）

#### 3.1 POPE 评估（已有脚本）
对所有 Phase 2 重训的 DPO 模型跑 POPE 3-split 评估。
```bash
python eval/generate_pope_answers.py --model_path MODEL --adapter_path ADAPTER --pope_dir data/pope_data --output_dir results/eval/NAME
python eval/eval_pope.py --results_dir results/eval/NAME
```

#### 3.2 CHAIR 评估（已有脚本）
对关键模型（baseline, true optimal, dpo-only）跑 CHAIR。
```bash
python eval/generate_chair_captions.py --model_path MODEL --adapter_path ADAPTER --image_dir COCO_DIR --num_images 500 --output_file results/eval/NAME/chair_captions.json
python eval/eval_chair.py --caption_file ... --annotation_file ... --output_file ...
```

#### 3.3 MMBench 评估（需新写脚本）
**需要新增**:
- `eval/generate_mmbench_answers.py` — 加载模型，处理 MMBench 问题，生成预测
- `eval/eval_mmbench.py` — 计算 MMBench 指标

MMBench 格式: 多选题 (A/B/C/D)，涵盖 20+ 能力维度。
评估 4 个关键模型: base, sft5k, dpo_true_optimal(RLAIF-V), dpo_only(RLAIF-V)

### Phase 4: 报告更新（本地，无需 GPU）

#### 4.1 更新实验数据
- 替换所有 DPO 相关的实验数据（POPE/CHAIR 表格）
- 新增 MMBench 结果章节/表格
- 更新方法论中数据集描述（RLHF-V → RLAIF-V）

#### 4.2 更新图表
- 重新生成包含 RLAIF-V 结果的对比图
- 新增 MMBench 能力雷达图
- 更新 DPO-only paradox 等关键图表

#### 4.3 更新叙事
- 核心发现是否仍然成立？（RLAIF-V 83K vs RLHF-V 5.7K 可能改变结论）
- "less is more" 结论不受影响（SFT 部分不变）
- DPO beta/loss/epoch 趋势可能变化

---

## 服务器选择

### 当前可用资源

| 服务器 | 可用 GPU | 空闲显存 | 模型 | 数据 | 推荐 |
|--------|---------|---------|------|------|------|
| 236 (215.236) | 无 | 全部高负载 | ✅ 有 | ✅ 全部 | ❌ 无空闲 GPU |
| 37 (234.37) | GPU 1 (~24GB) | 勉强 | ✅ 有 8B | 部分 | ⚠️ 仅够 eval |

**问题**: 两台服务器目前都没有足够的空闲 GPU 做 DPO 训练（需要 2 张各 25GB+ 空闲的卡）。

**解决方案**:
1. 等 GPU 释放（监控 nvidia-smi）
2. 和同学协调 GPU 使用时间
3. 使用单卡训练（增大 gradient accumulation 补偿）

---

## 关键风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| RLAIF-V 83K 训练时间过长 | DPO 从 ~3h 增加到 ~15-20h | 限制 max_samples (如 20K) |
| GPU 不可用 | 无法训练 | 监控等待，或用单卡 |
| RLAIF-V 改变核心结论 | 报告需大幅修改 | 先跑 baseline 确认趋势 |
| MMBench 脚本编写 | 额外开发工作 | 参考 lmms-eval 或 opencompass |
| 服务器无外网 | 无法下载数据 | 本地下载后 scp |

---

## 时间线估算

| 阶段 | 工作内容 | 耗时 | 前置条件 |
|------|---------|------|---------|
| Phase 1 | 数据准备 + SCP 上传 | 2-3h | 无 |
| Phase 2 | DPO 重训练 (P0 实验) | 8-12h GPU | GPU 空闲 |
| Phase 3 | 评估 (POPE+CHAIR+MMBench) | 5-8h GPU | Phase 2 完成 |
| Phase 4 | 报告更新 | 3-5h | Phase 3 完成 |
| **总计** | | **~20-30h** | |

---

## 保留的工作（无需修改）

- ✅ SFT 训练（LLaVA-Instruct-150K 符合课程要求）
- ✅ POPE 评估脚本和流程
- ✅ CHAIR 评估脚本和流程
- ✅ 所有 SFT 消融实验结果
- ✅ 报告框架和大部分章节
- ✅ Case study 方法论
