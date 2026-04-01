# 实验接力执行方案

> 本文档供自动监控使用，每 5 分钟检查一次实验状态，任务完成后自动启动下一个。

## 服务器信息
- SSH: `ssh research@115.190.215.236`
- 项目目录: `/mnt/disk2/lijunlin/vqa-hallucination`
- Conda: `export PATH=/mnt/disk3/conda/miniconda3/envs/zh/bin:$PATH`
- LD_LIBRARY_PATH: `export LD_LIBRARY_PATH=/mnt/disk3/conda/miniconda3/envs/zh/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH`
- 模型路径: `../downloads/models/Qwen3-VL-8B-Instruct`
- COCO val2014: `../downloads/coco/val2014`
- CHAIR 标注: `data/coco_val2014_chair_annots.json`

## GPU 占用规则
- **绝不挤爆共享 GPU**，启动前先 `nvidia-smi` 检查
- SFT 单卡训练: ~46GB（需要一整张空闲卡，或与 <35GB 进程共存）
- DPO DDP 双卡: ~24GB/卡（需要两张各有 >25GB 空闲的卡）
- POPE 评估: ~18GB（可与 <60GB 进程共存）
- CHAIR caption 生成: ~18GB（同上）
- 所有训练配置都有 `save_steps` + `save_total_limit`，支持断点续跑

## eval_pope.py 重要修复
`parse_yesno` 已修复，能处理 `<think>...</think>` 前缀。**同步到服务器的版本已是修复版。**

## 启动命令模板

### 训练（SFT 单卡）
```bash
CUDA_VISIBLE_DEVICES=X nohup llamafactory-cli train configs/CONFIG.yaml > logs/ablation/NAME.log 2>&1 &
```

### 训练（DPO 双卡 DDP）
```bash
CUDA_VISIBLE_DEVICES=X,Y FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 nohup llamafactory-cli train configs/CONFIG.yaml > logs/ablation/NAME.log 2>&1 &
```

### POPE 评估
```bash
CUDA_VISIBLE_DEVICES=X nohup python eval/generate_pope_answers.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path ADAPTER_PATH \
    --pope_dir data/pope_data \
    --output_dir results/eval/NAME \
    > logs/eval_NAME.log 2>&1 &
```
（Base 模型不加 `--adapter_path`）

### CHAIR caption 生成
```bash
CUDA_VISIBLE_DEVICES=X nohup python eval/generate_chair_captions.py \
    --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
    --adapter_path ADAPTER_PATH \
    --image_dir ../downloads/coco/val2014 \
    --num_images 500 \
    --output_file results/eval/NAME/chair_captions.json \
    > logs/chair_NAME.log 2>&1 &
```

### CHAIR 评估（不需要 GPU）
```bash
python eval/eval_chair.py \
    --caption_file results/eval/NAME/chair_captions.json \
    --annotation_file data/coco_val2014_chair_annots.json \
    --output_file results/eval/NAME/chair_metrics.json
```

---

## 当前运行中的实验（截至 2026-03-29 00:15）

| 实验 | GPU | 进度 | 日志 |
|------|-----|------|------|
| DPO eval (SFT+DPO) | GPU 3 | pope_adversarial ~26% | logs/eval_dpo.log |
| DPO beta=0.05 | GPU 4,5 | ~38% | logs/ablation/dpo_beta005.log |
| SFT r=4 | GPU 1 | ~33% | logs/ablation/sft_r4.log |
| SFT r=16 | GPU 6 | ~14% | logs/ablation/sft_r16.log |
| SFT r=32 | GPU 7 | ~14% | logs/ablation/sft_r32.log |

---

## 接力任务队列（按优先级）

### 当 GPU 3 空出（DPO eval 完成后，约 1h）：
1. `POPE eval: DPO-only` → adapter: `results/dpo/dpo_only_r8`, output: `results/eval/dpo_only`
2. `CHAIR: Base` → 无 adapter, output: `results/eval/base/chair_captions.json`
3. `CHAIR: SFT` → adapter: `results/sft/lora_r8`, output: `results/eval/sft/chair_captions.json`
4. `CHAIR: SFT+DPO` → adapter: `results/dpo/lora_r8_beta01`, output: `results/eval/dpo/chair_captions.json`

### 当 GPU 4,5 空出（DPO beta=0.05 完成后，约 1.5h）：
5. `DPO beta=0.2` → config: `qwen3vl_dpo_beta02.yaml`, log: `dpo_beta02.log`
6. `DPO beta=0.5` → config: `qwen3vl_dpo_beta05.yaml`, log: `dpo_beta05.log`
7. `DPO hinge` → config: `qwen3vl_dpo_loss_hinge.yaml`, log: `dpo_hinge.log`
8. `DPO IPO` → config: `qwen3vl_dpo_loss_ipo.yaml`, log: `dpo_ipo.log`

### 当 GPU 1 空出（SFT r=4 完成后，约 4-5h）：
9. `SFT data=5K` → config: `qwen3vl_sft_data5k.yaml`, log: `sft_data5k.log`（~30min）
10. `SFT data=10K` → config: `qwen3vl_sft_data10k.yaml`, log: `sft_data10k.log`（~1h）
11. `SFT data=25K` → config: `qwen3vl_sft_data25k.yaml`, log: `sft_data25k.log`（~2.5h）

### 当 GPU 6 或 7 空出（SFT r16/r32 完成后，约 7-8h）：
12. 批量 POPE 评估所有已完成的消融模型（每个 ~1h，单卡 ~18GB）

消融模型 POPE 评估清单：
- `results/ablation/sft_r4` → `results/eval/ablation_sft_r4`
- `results/ablation/sft_lora_r16` → `results/eval/ablation_sft_r16`
- `results/ablation/sft_lora_r32` → `results/eval/ablation_sft_r32`
- `results/ablation/dpo_beta001` → `results/eval/ablation_dpo_beta001`
- `results/ablation/dpo_beta005` → `results/eval/ablation_dpo_beta005`
- `results/ablation/dpo_beta02` → `results/eval/ablation_dpo_beta02`
- `results/ablation/dpo_beta05` → `results/eval/ablation_dpo_beta05`
- `results/ablation/dpo_loss_hinge` → `results/eval/ablation_dpo_hinge`
- `results/ablation/dpo_loss_ipo` → `results/eval/ablation_dpo_ipo`
- `results/ablation/sft_data5k` → `results/eval/ablation_sft_data5k`
- `results/ablation/sft_data10k` → `results/eval/ablation_sft_data10k`
- `results/ablation/sft_data25k` → `results/eval/ablation_sft_data25k`

注意：DPO 消融模型的 adapter 路径需要确认（可能在 `results/ablation/XXX` 下）。

### 全部完成后：
13. 汇总分析出图（本地执行）：
    - `python eval/eval_compare.py`
    - `python eval/compare_training_curves.py`
    - `python eval/generate_case_study.py`

---

## 已完成的实验

- [x] SFT baseline: LoRA r=8, 50K, 2 epochs → `results/sft/lora_r8`
- [x] DPO baseline: beta=0.1, sigmoid → `results/dpo/lora_r8_beta01`
- [x] DPO-only: 直接在 base 上 DPO → `results/dpo/dpo_only_r8`
- [x] Base POPE eval → `results/eval/base/` (3 splits ✓)
- [x] SFT POPE eval → `results/eval/sft/` (3 splits ✓)
- [x] DPO beta=0.01 → `results/ablation/dpo_beta001`

## 已知问题
- 服务器无法访问国外站点，所有外部数据需本地下载后 scp
- SFT 模型输出带 `<think>` 标签，eval_pope.py 已修复
- GPU 0,2 被其他进程长期占用（VLLM, llm_ad），不要使用
- GPU 1 有 llm_ad (20GB) 共存，剩余空间够 SFT 单卡
