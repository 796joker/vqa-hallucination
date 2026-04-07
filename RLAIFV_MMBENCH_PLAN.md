# RLAIF-V + MMBench 补充实验计划

> 创建日期: 2026-04-06
> 最后更新: 2026-04-07 17:00
> 服务器: 236 (ssh research@115.190.215.236)
> 目标: 使用课程推荐数据集(RLAIF-V)和评估集(MMBench)补充实验
> **状态: 全部完成 ✅**

---

## 当前执行状态

### Phase 0 数据准备 ✅ 已完成
- [x] RLAIF-V parquets 下载 (12GB, 83K pairs)
- [x] 转换为 LLaMA-Factory 格式 (20K samples → `data/dpo_data/rlaifv_dpo.json`)
- [x] 图片提取 (20K JPEGs → `data/dpo_data/rlaifv_images/`)
- [x] 6 个训练配置文件创建
- [x] 图片路径修复 (去除 `data/` 前缀适配 LLaMA-Factory media_dir)

### Phase 1 DPO 训练 ✅ 已完成 (2026-04-06 18:32 — 2026-04-07 05:47)

| 实验 | 配置 | GPU | 总步数 | 当前进度 | 速度 | 预计完成 |
|------|------|-----|--------|---------|------|---------|
| **baseline** | SFT50K+β=0.1+3ep | 3,4 DDP | 3750 | 106/3750 (2.8%) | ~14s/step | 4/7 ~08:00 |
| **optimal** | SFT5K+β=1.0+1ep | 5,6 DDP | 1250 | 105/1250 (8.4%) | ~13s/step | 4/6 ~23:00 |
| **epoch1** | SFT50K+β=0.1+1ep | 0,1 DDP | 1250 | 27/1250 (2.2%) | ~11s/step | 4/6 ~23:00 |
| **only** | Base+β=0.1+3ep | 7 单卡 | 待定 | tokenize 91% | - | tokenize后~16h |

**GPU 分配说明**:
- GPU 3,4,5,6,7: 独占（无其他进程）
- GPU 0,1: 与他人共享（memory_baseline 6.4GB + mi-persuasion 32.1+23.6GB），显存余量 20+35GB 安全
- GPU 2: 未使用（VLLM 72GB 占满）

**数据说明**:
- 当前使用 RLAIF-V 83K 中的 20K 子集 (vs RLHF-V 完整 5.7K)
- 规模差异 (20K vs 5.7K) 意味着对比结论受数据规模和标注质量两个变量影响

### 计划中：5.7K RLAIF-V 控制实验

为消除规模变量、纯粹比较**人工标注 vs AI 标注**质量差异，计划增加一组：
- 从 RLAIF-V 83K 中随机采样 **5733 条**（与 RLHF-V 完全相同规模）
- 使用 optimal 配置 (SFT5K + β=1.0 + 1ep) 训练
- 对比 RLHF-V 5.7K optimal (F1=0.889, CHAIR_i=20.12%) vs RLAIF-V 5.7K optimal

需要：
1. 生成 `data/dpo_data/rlaifv_dpo_5k.json`（从 rlaifv_dpo.json 随机采样 5733 条）
2. 在 `dataset_info.json` 中新增 `rlaifv_dpo_5k` 条目
3. 新建配置 `qwen3vl_dpo_rlaifv_optimal_5k.yaml`（dataset=rlaifv_dpo_5k，其余同 optimal）
4. 等当前训练释放 GPU 后启动

**价值**：这是唯一能回答"RLHF-V 人工标注 vs RLAIF-V AI 标注在同等规模下谁更好"的实验，对报告的说服力有显著提升。

### Phase 2 评估 ✅ 已完成 (POPE + CHAIR + MMBench)

### Phase 3 MMBench 脚本 ✅ 已完成

**MMBench 结果 (2026-04-07)**:
| 模型 | MMBench Acc |
|------|-----------|
| Base | 89.72% |
| SFT 5K | 89.35% |
| RLAIF-V optimal_5k | 89.47% |
| RLAIF-V only | 89.65% |

---

## 一、前置条件

### 1.1 数据已就绪
- [x] RLAIF-V: `/mnt/disk2/lijunlin/downloads/datasets/RLAIF-V/` (12GB, 14 parquets, ~83K pairs)
- [x] MMBench: `/mnt/disk2/lijunlin/downloads/datasets/MMBench/` (503MB, en/cn/cc splits)

### 1.2 环境
```bash
export PATH=/mnt/disk3/conda/miniconda3/envs/zh/bin:$PATH
export LD_LIBRARY_PATH=/mnt/disk3/conda/miniconda3/envs/zh/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
cd /mnt/disk2/lijunlin/vqa-hallucination
```

### 1.3 已有资源（无需重做）
- SFT 5K adapter: `results/ablation/sft_data5k/` (True Optimal 的 SFT 基础)
- SFT 50K adapter: `results/sft/lora_r8/` (baseline SFT)
- 基座模型: `../downloads/models/Qwen3-VL-8B-Instruct`
- COCO val2014: `../downloads/coco/val2014`

---

## 二、数据准备（Phase 0，无需 GPU）

### 2.1 转换 RLAIF-V 为 LLaMA-Factory 格式

项目中已有转换脚本 `data/prepare_dpo_data.py`，需要适配 parquet 格式：

```bash
python data/prepare_dpo_data.py \
    --input_dir /mnt/disk2/lijunlin/downloads/datasets/RLAIF-V/ \
    --output data/dpo_data/rlaifv_dpo.json \
    --max_samples 83000
```

如果脚本不支持 parquet 输入，需要先用 Python 转换：

```python
import pandas as pd
import json, glob

parquets = sorted(glob.glob("/mnt/disk2/lijunlin/downloads/datasets/RLAIF-V/*.parquet"))
dfs = [pd.read_parquet(p) for p in parquets]
df = pd.concat(dfs)
print(f"Total samples: {len(df)}")
print(f"Columns: {list(df.columns)}")
# 转换为 LLaMA-Factory sharegpt ranking 格式并保存
```

转换后更新 `data/dataset_info.json`，确保 `rlaifv_dpo` 条目的 `file_name` 指向 `dpo_data/rlaifv_dpo.json`。

### 2.2 准备 MMBench 评估数据

```bash
# 解析 MMBench parquet 为评估脚本可用的格式
python data/prepare_mmbench.py \
    --input_dir /mnt/disk2/lijunlin/downloads/datasets/MMBench/ \
    --output_dir data/mmbench/
```

需要新写 `data/prepare_mmbench.py` 和 `eval/generate_mmbench_answers.py` + `eval/eval_mmbench.py`。

MMBench 格式: 多选题 (A/B/C/D)，需要从模型输出中提取选项字母。

---

## 三、DPO 训练计划（Phase 1，需要 GPU）

### 3.1 需要新建的配置文件

所有 RLAIF-V DPO 配置与 RLHF-V 版本相同，仅改 `dataset: rlaifv_dpo`。
由于 RLAIF-V 有 83K pairs (vs RLHF-V 5.7K)，训练时间会增加 ~14x。
**建议**: 使用 `max_samples` 限制到 20K，平衡时间和效果。

#### 配置文件清单

| 配置 | SFT 基础 | Beta | Loss | Epochs | 预估时间 (DDP 2卡) | 优先级 |
|------|---------|------|------|--------|-------------------|--------|
| `qwen3vl_dpo_rlaifv_baseline.yaml` | SFT 50K | 0.1 | sigmoid | 3 | ~8h | P0 |
| `qwen3vl_dpo_rlaifv_optimal.yaml` | SFT 5K | 1.0 | sigmoid | 1 | ~3h | P0 |
| `qwen3vl_dpo_rlaifv_only.yaml` | 无 (base) | 0.1 | sigmoid | 3 | ~8h | P0 |
| `qwen3vl_dpo_rlaifv_beta05.yaml` | SFT 50K | 0.5 | sigmoid | 3 | ~8h | P1 |
| `qwen3vl_dpo_rlaifv_beta10.yaml` | SFT 50K | 1.0 | sigmoid | 3 | ~8h | P1 |
| `qwen3vl_dpo_rlaifv_epoch1.yaml` | SFT 50K | 0.1 | sigmoid | 1 | ~3h | P1 |

### 3.2 配置文件模板

以 `qwen3vl_dpo_rlaifv_baseline.yaml` 为例（基于现有 `qwen3vl_dpo_lora.yaml` 修改）:

```yaml
# 修改点:
# 1. dataset: rlaifv_dpo (替代 rlhf_v)
# 2. max_samples: 20000 (限制训练规模)
# 3. output_dir: results/ablation/dpo_rlaifv_baseline

model_name_or_path: ../downloads/models/Qwen3-VL-8B-Instruct
adapter_name_or_path: results/sft/lora_r8
finetuning_type: lora
stage: dpo
dataset: rlaifv_dpo
max_samples: 20000
# ... 其余参数与 qwen3vl_dpo_lora.yaml 相同
output_dir: results/ablation/dpo_rlaifv_baseline
```

### 3.3 GPU 需求

| 训练方式 | 显存需求 | 所需卡数 |
|---------|---------|---------|
| DPO DDP | ~24GB/卡 | 2 张，各需 >25GB 空闲 |
| DPO 单卡 | ~46GB | 1 张，需 >48GB 空闲 |

**当前 236 GPU 状态**: 全部高负载，无空闲。需等待 GPU 释放。
**当前 37 GPU 状态**: GPU 1 有 ~24GB 空闲，不够单独训练；无 conda 环境。

---

## 四、评估计划（Phase 2，需要 GPU）

### 4.1 POPE + CHAIR 评估（已有脚本）

对所有 Phase 1 训练的 RLAIF-V DPO 模型执行：

```bash
# POPE 评估 (~1h/model, ~18GB)
CUDA_VISIBLE_DEVICES=X python eval/generate_pope_answers.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_rlaifv_NAME \
    --pope_dir data/pope_data \
    --output_dir results/eval/dpo_rlaifv_NAME

# CHAIR 评估 (~1h/model, ~18GB)
CUDA_VISIBLE_DEVICES=X python eval/generate_chair_captions.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_rlaifv_NAME \
    --image_dir ../downloads/coco/val2014 \
    --num_images 500 \
    --output_file results/eval/dpo_rlaifv_NAME/chair_captions.json

python eval/eval_chair.py \
    --caption_file results/eval/dpo_rlaifv_NAME/chair_captions.json \
    --annotation_file data/coco_val2014_chair_annots.json \
    --output_file results/eval/dpo_rlaifv_NAME/chair_results.json
```

### 4.2 MMBench 评估（需新写脚本）

**需要新增的文件**:
- `eval/generate_mmbench_answers.py` — 加载模型，处理 MMBench 多选题，生成预测
- `eval/eval_mmbench.py` — 计算 MMBench 指标（支持 CircularEval）

MMBench 评估 4 个关键模型:
1. Base (无 adapter)
2. SFT 5K
3. DPO RLAIF-V optimal (SFT 5K + β=1.0 + 1ep)
4. DPO RLAIF-V only

```bash
# MMBench 评估 (~1h/model, ~18GB)
CUDA_VISIBLE_DEVICES=X python eval/generate_mmbench_answers.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path ADAPTER \
    --mmbench_dir data/mmbench \
    --output_dir results/eval/MODEL_NAME
```

---

## 五、并行执行策略

### 5.1 训练阶段并行

所有 DPO 训练使用 DDP 多卡并行（与之前实验相同方式）：

```bash
# DDP 2卡训练
CUDA_VISIBLE_DEVICES=X,Y FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
    nohup llamafactory-cli train configs/qwen3vl_dpo_rlaifv_NAME.yaml \
    > logs/dpo_rlaifv_NAME.log 2>&1 &
```

如果有 3+ 张卡空闲，可以 3 卡 DDP 进一步加速（之前 true_optimal 用过 3 卡）：

```bash
# DDP 3卡训练 (更快)
CUDA_VISIBLE_DEVICES=X,Y,Z FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=3 \
    nohup llamafactory-cli train configs/qwen3vl_dpo_rlaifv_NAME.yaml \
    > logs/dpo_rlaifv_NAME.log 2>&1 &
```

### 5.2 评估阶段并行

评估为单卡任务（~18GB），多个评估可在不同卡上同时运行：

```bash
# 同时在不同卡上跑多个评估
CUDA_VISIBLE_DEVICES=X nohup python eval/generate_pope_answers.py ... > logs/eval1.log 2>&1 &
CUDA_VISIBLE_DEVICES=Y nohup python eval/generate_pope_answers.py ... > logs/eval2.log 2>&1 &
CUDA_VISIBLE_DEVICES=Z nohup python eval/generate_chair_captions.py ... > logs/eval3.log 2>&1 &
```

### 5.3 执行接力队列

当 GPU 空出时按以下优先级执行：

**Phase 0 (无 GPU, 立即可做)**:
1. 转换 RLAIF-V 数据为 LLaMA-Factory 格式
2. 准备 MMBench 评估数据
3. 编写 MMBench 评估脚本
4. 创建所有 DPO 配置文件

**Phase 1 (需 2+ 张 GPU, ~25GB/卡)**:
1. P0: DPO RLAIF-V baseline (SFT50K + β=0.1 + 3ep)
2. P0: DPO RLAIF-V optimal (SFT5K + β=1.0 + 1ep)
3. P0: DPO RLAIF-V only (base + β=0.1 + 3ep)

**Phase 2 (需 1 张 GPU, ~18GB, 可多卡并行)**:
1. POPE 评估: 所有 Phase 1 模型 (3-6 个)
2. CHAIR 评估: P0 模型 (3 个)
3. MMBench 评估: 4 个关键模型 (base + sft5k + rlaifv_optimal + rlaifv_only)

---

## 六、预期对比分析

完成后将产出以下对比：

### 6.1 RLHF-V vs RLAIF-V 对比（核心）

| 模型 | POPE F1 | CHAIR_i | 训练数据 |
|------|---------|---------|---------|
| DPO baseline (RLHF-V 5.7K) | 0.780 | 18.88% | 人工标注 |
| DPO baseline (RLAIF-V 20K) | ? | ? | AI 标注 |
| DPO optimal (RLHF-V) | 0.889 | 20.12% | 人工标注 |
| DPO optimal (RLAIF-V) | ? | ? | AI 标注 |

**关键问题**: RLAIF-V 的 14x 规模优势能否弥补人工→AI 标注的质量差距？

### 6.2 MME vs MMBench 对比

| 模型 | MME Total | MMBench Acc | 评估方式 |
|------|----------|-------------|---------|
| Base | 2008.0 | ? | yes/no vs 多选 |
| SFT 5K | 1899.0 | ? | |
| True Optimal (RLHF-V) | 1990.5 | ? | |
| True Optimal (RLAIF-V) | ? | ? | |

**关键问题**: MMBench 的 CircularEval 是否揭示 MME 未发现的能力差异？

---

## 七、时间估算

| 阶段 | 工作内容 | 耗时 | GPU 需求 |
|------|---------|------|---------|
| Phase 0 | 数据转换 + 脚本编写 | 2-3h | 无 |
| Phase 1 | P0 DPO 训练 (3 个) | 8-12h | 2卡 × 25GB |
| Phase 1 | P1 DPO 训练 (3 个, 可选) | 8-12h | 2卡 × 25GB |
| Phase 2 | POPE+CHAIR 评估 | 6-8h | 1卡 × 18GB (可并行) |
| Phase 2 | MMBench 评估 | 4-5h | 1卡 × 18GB |
| **总计** | | **~25-35h GPU** | |

---

## 八、启动检查清单

在 GPU 空闲时按此清单逐步执行：

- [ ] `nvidia-smi` 确认至少 2 张卡各有 >25GB 空闲
- [ ] Phase 0 数据准备完成（RLAIF-V 转换 + MMBench 准备）
- [ ] 配置文件已创建并检查
- [ ] 启动第一个 P0 训练任务
- [ ] 设置监控: `watch -n 60 nvidia-smi`
- [ ] 训练完成后立即启动 POPE+CHAIR 评估
- [ ] MMBench 脚本就绪后启动 MMBench 评估
