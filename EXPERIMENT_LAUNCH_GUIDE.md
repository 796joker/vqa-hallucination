# 实验启动指南

> 服务器: ssh research@115.190.215.236
> 项目目录: /mnt/disk2/lijunlin/vqa-hallucination
> Conda 环境: zh (/mnt/disk3/conda/miniconda3/envs/zh/bin)

---

## 零、环境初始化

每次登录服务器后必须先执行：

```bash
export PATH=/mnt/disk3/conda/miniconda3/envs/zh/bin:$PATH
export LD_LIBRARY_PATH=/mnt/disk3/conda/miniconda3/envs/zh/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
cd /mnt/disk2/lijunlin/vqa-hallucination
```

启动前必须检查 GPU：
```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

---

## 一、RLHF-V DPO 训练（原有实验）

### 显存需求
- DDP 2 卡: ~24GB/卡
- DDP 3 卡: ~22GB/卡
- 单卡: ~46GB

### Baseline (SFT 50K + beta=0.1 + 3ep)
```bash
CUDA_VISIBLE_DEVICES=X,Y FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
    nohup llamafactory-cli train configs/qwen3vl_dpo_lora.yaml \
    > logs/dpo_baseline.log 2>&1 &
```

### True Optimal (SFT 5K + beta=1.0 + 1ep)
```bash
CUDA_VISIBLE_DEVICES=X,Y FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
    nohup llamafactory-cli train configs/qwen3vl_dpo_true_optimal.yaml \
    > logs/dpo_true_optimal.log 2>&1 &
```

### DPO-only (Base + beta=0.1 + 3ep)
```bash
CUDA_VISIBLE_DEVICES=X,Y FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
    nohup llamafactory-cli train configs/qwen3vl_dpo_only.yaml \
    > logs/dpo_only.log 2>&1 &
```

### 其他消融 (替换对应 config)
- beta 消融: `qwen3vl_dpo_beta{001,005,02,05,10}.yaml`
- loss 消融: `qwen3vl_dpo_loss_{hinge,ipo}.yaml`
- epoch 消融: `qwen3vl_dpo_epoch1.yaml`
- optimal: `qwen3vl_dpo_optimal.yaml`

---

## 二、RLAIF-V DPO 训练（补充实验）

### 数据准备（Phase 0, 无需 GPU, 已完成 ✅）

```bash
# 转换 RLAIF-V parquets -> LLaMA-Factory JSON (20K samples)
python3 data/prepare_dpo_data.py \
    --input_dir /mnt/disk2/lijunlin/downloads/datasets/RLAIF-V/ \
    --output data/dpo_data/rlaifv_dpo.json \
    --image_dir data/dpo_data/rlaifv_images \
    --max_samples 20000
```

输出:
- `data/dpo_data/rlaifv_dpo.json` (20MB, 20000 samples)
- `data/dpo_data/rlaifv_images/` (20000 张 JPEG)

### 训练启动

所有配置与 RLHF-V 版本仅 `dataset` 和 `output_dir` 不同。

#### P0 优先级（并行启动）

```bash
# 1. Baseline: SFT50K + RLAIF-V + beta=0.1 + 3ep (GPU X,Y DDP)
CUDA_VISIBLE_DEVICES=3,4 FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
    nohup llamafactory-cli train configs/qwen3vl_dpo_rlaifv_baseline.yaml \
    > logs/dpo_rlaifv_baseline.log 2>&1 &

# 2. Optimal: SFT5K + RLAIF-V + beta=1.0 + 1ep (GPU X,Y DDP)
CUDA_VISIBLE_DEVICES=5,6 FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
    nohup llamafactory-cli train configs/qwen3vl_dpo_rlaifv_optimal.yaml \
    > logs/dpo_rlaifv_optimal.log 2>&1 &

# 3. DPO-only: Base + RLAIF-V + beta=0.1 + 3ep (GPU Z 单卡)
CUDA_VISIBLE_DEVICES=7 \
    nohup llamafactory-cli train configs/qwen3vl_dpo_rlaifv_only.yaml \
    > logs/dpo_rlaifv_only.log 2>&1 &
```

#### P1 优先级（P0 完成后启动）
```bash
# 4. Beta=0.5: SFT50K + RLAIF-V + beta=0.5 + 3ep
CUDA_VISIBLE_DEVICES=X,Y FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
    nohup llamafactory-cli train configs/qwen3vl_dpo_rlaifv_beta05.yaml \
    > logs/dpo_rlaifv_beta05.log 2>&1 &

# 5. Beta=1.0: SFT50K + RLAIF-V + beta=1.0 + 3ep
CUDA_VISIBLE_DEVICES=X,Y FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
    nohup llamafactory-cli train configs/qwen3vl_dpo_rlaifv_beta10.yaml \
    > logs/dpo_rlaifv_beta10.log 2>&1 &

# 6. Epoch=1: SFT50K + RLAIF-V + beta=0.1 + 1ep
CUDA_VISIBLE_DEVICES=X,Y FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
    nohup llamafactory-cli train configs/qwen3vl_dpo_rlaifv_epoch1.yaml \
    > logs/dpo_rlaifv_epoch1.log 2>&1 &
```

### 配置对照表

| 配置文件 | SFT 基础 | Beta | Epochs | 数据集 | 输出目录 |
|---------|---------|------|--------|-------|---------|
| `qwen3vl_dpo_rlaifv_baseline.yaml` | SFT 50K | 0.1 | 3 | rlaifv_dpo | results/ablation/dpo_rlaifv_baseline |
| `qwen3vl_dpo_rlaifv_optimal.yaml` | SFT 5K | 1.0 | 1 | rlaifv_dpo | results/ablation/dpo_rlaifv_optimal |
| `qwen3vl_dpo_rlaifv_only.yaml` | 无 (base) | 0.1 | 3 | rlaifv_dpo | results/ablation/dpo_rlaifv_only |
| `qwen3vl_dpo_rlaifv_beta05.yaml` | SFT 50K | 0.5 | 3 | rlaifv_dpo | results/ablation/dpo_rlaifv_beta05 |
| `qwen3vl_dpo_rlaifv_beta10.yaml` | SFT 50K | 1.0 | 3 | rlaifv_dpo | results/ablation/dpo_rlaifv_beta10 |
| `qwen3vl_dpo_rlaifv_epoch1.yaml` | SFT 50K | 0.1 | 1 | rlaifv_dpo | results/ablation/dpo_rlaifv_epoch1 |

---

## 三、SFT 训练（原有实验，已全部完成）

```bash
# 单卡 SFT (~46GB)
CUDA_VISIBLE_DEVICES=X \
    nohup llamafactory-cli train configs/qwen3vl_sft_lora.yaml \
    > logs/sft_baseline.log 2>&1 &
```

消融配置: `qwen3vl_sft_{data5k,data10k,data25k,lora_r4,lora_r16,lora_r32}.yaml`

---

## 四、POPE 评估

### 显存需求: ~18GB/模型

```bash
# 需要 adapter 的模型
CUDA_VISIBLE_DEVICES=X nohup python eval/generate_pope_answers.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path ADAPTER_DIR \
    --pope_dir data/pope_data \
    --output_dir results/eval/OUTPUT_NAME \
    > logs/eval_pope_NAME.log 2>&1 &

# Base 模型 (无 adapter)
CUDA_VISIBLE_DEVICES=X nohup python eval/generate_pope_answers.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --pope_dir data/pope_data \
    --output_dir results/eval/base \
    > logs/eval_pope_base.log 2>&1 &
```

### 评分 (无需 GPU)
```bash
python eval/eval_pope.py --results_dir results/eval/OUTPUT_NAME
```

### RLAIF-V 模型评估命令
```bash
# 可多卡并行，每个 eval 占 1 卡 ~18GB
CUDA_VISIBLE_DEVICES=3 nohup python eval/generate_pope_answers.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_rlaifv_baseline \
    --pope_dir data/pope_data \
    --output_dir results/eval/dpo_rlaifv_baseline \
    > logs/eval_pope_rlaifv_baseline.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python eval/generate_pope_answers.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_rlaifv_optimal \
    --pope_dir data/pope_data \
    --output_dir results/eval/dpo_rlaifv_optimal \
    > logs/eval_pope_rlaifv_optimal.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python eval/generate_pope_answers.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_rlaifv_only \
    --pope_dir data/pope_data \
    --output_dir results/eval/dpo_rlaifv_only \
    > logs/eval_pope_rlaifv_only.log 2>&1 &
```

---

## 五、CHAIR 评估

### Caption 生成 (需 GPU, ~18GB)
```bash
CUDA_VISIBLE_DEVICES=X nohup python eval/generate_chair_captions.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path ADAPTER_DIR \
    --image_dir ../downloads/coco/val2014 \
    --num_images 500 \
    --output_file results/eval/OUTPUT_NAME/chair_captions.json \
    > logs/chair_NAME.log 2>&1 &
```

### 评分 (无需 GPU)
```bash
python eval/eval_chair.py \
    --caption_file results/eval/OUTPUT_NAME/chair_captions.json \
    --annotation_file data/coco_val2014_chair_annots.json \
    --output_file results/eval/OUTPUT_NAME/chair_results.json
```

---

## 六、MME 评估（原有实验）

```bash
# 生成答案 (需 GPU, ~18GB)
CUDA_VISIBLE_DEVICES=X nohup python eval/generate_mme_answers.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path ADAPTER_DIR \
    --mme_dir data/mme \
    --output_dir results/eval/OUTPUT_NAME \
    > logs/mme_NAME.log 2>&1 &

# 评分 (无需 GPU)
python eval/eval_mme.py --results_dir results/eval/OUTPUT_NAME
```

---

## 七、Gradio Demo

### 显存需求: ~32GB (两个模型 bf16) 或 ~16GB (4-bit 量化)

```bash
# bf16 (推荐, 需 >=40GB 空闲 GPU)
CUDA_VISIBLE_DEVICES=X python demo/app.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_true_optimal \
    --port 7860 --share

# 4-bit 量化 (显存不足时)
CUDA_VISIBLE_DEVICES=X python demo/app.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_true_optimal \
    --use_4bit --port 7860 --share
```

---

## 八、监控命令

```bash
# GPU 占用
watch -n 10 nvidia-smi

# 训练日志
tail -f logs/dpo_rlaifv_baseline.log

# 检查训练进度 (loss, steps)
grep "loss" logs/dpo_rlaifv_baseline.log | tail -5

# 检查进程
ps aux | grep llamafactory | grep -v grep
```

---

## 九、安全规则

1. **启动前必须 nvidia-smi** — 确认目标 GPU 有足够空闲显存
2. **绝不触碰其他人的进程** — GPU 0 (memory_baseline + mi-persuasion), GPU 2 (VLLM)
3. **所有训练用 nohup** — SSH 断开不影响训练
4. **save_steps + save_total_limit** — 所有配置都有断点续跑支持
5. **只 kill 自己的进程** — 确认 PID 属于 `zh` 环境或 `llamafactory` 关键字
