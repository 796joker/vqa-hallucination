# Gradio Demo 部署指南（AutoDL / FunHPC）

> 用于课程展示，在租卡平台上运行 Gradio Demo

---

## 一、GPU 选型

### 显存需求

| 模式 | 加载内容 | 显存需求 | 推荐卡型 |
|------|---------|---------|---------|
| **bf16（推荐）** | Base 模型 ~16GB + Fine-tuned 模型 ~16GB | **~32GB** | A100-40GB, A6000-48GB |
| **4-bit 量化** | Base ~5GB + Fine-tuned ~5GB | **~12GB** | RTX 4090-24GB, A5000-24GB |

### AutoDL 推荐选择

| 优先级 | 卡型 | 显存 | 价格参考 | 说明 |
|--------|------|------|---------|------|
| 1 | **A100-SXM4-40GB** | 40GB | ~¥3/h | bf16 全精度，效果最好 |
| 2 | **A100-SXM4-80GB** | 80GB | ~¥5/h | 显存充裕但价格贵 |
| 3 | **A6000-48GB** | 48GB | ~¥2.5/h | 性价比高 |
| 4 | **RTX 4090-24GB** | 24GB | ~¥1.5/h | 需 4-bit 量化，便宜 |

> 展示时间约 15-30 分钟，总费用 ¥1-3。

---

## 二、镜像选择

### 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| **Python** | 3.10 - 3.12 | 3.12 最佳（与训练环境一致） |
| **CUDA** | 12.1+ | 12.4 最佳 |
| **PyTorch** | 2.4.0+ | 2.10.0 最佳（与训练一致），2.4+ 均可 |

### AutoDL 镜像推荐

1. **首选**: `PyTorch 2.4.0 + Python 3.12 + CUDA 12.4`
2. **备选**: `PyTorch 2.3.0 + Python 3.10 + CUDA 12.1`
3. 如果没有上述镜像，选 `PyTorch 2.x + CUDA 12.x` 的最新版

### FunHPC 镜像推荐

选择 `PyTorch` 框架，CUDA 12.x，Python 3.10+。

---

## 三、环境配置

登录实例后执行：

```bash
# 1. 安装依赖（约 3-5 分钟）
pip install transformers==4.57.6 peft==0.18.1 accelerate==1.11.0 \
    gradio==5.50.0 qwen-vl-utils==0.0.14 pillow==11.3.0

# 如果用 4-bit 量化，额外安装
pip install bitsandbytes

# 2. 验证安装
python -c "
import torch, transformers, peft, gradio, accelerate
print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'gradio: {gradio.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"
```

---

## 四、模型和数据下载

### 需要下载的文件

| 文件 | 大小 | 来源 | 说明 |
|------|------|------|------|
| Qwen3-VL-8B-Instruct | **17GB** | HuggingFace | 基座模型 |
| dpo_true_optimal adapter | **630MB** | GitHub 仓库 | LoRA 权重 |
| demo/app.py | 几KB | GitHub 仓库 | Demo 代码 |
| demo/examples/ | 几MB | GitHub 仓库 | 示例图片 |

### 下载步骤

```bash
# 1. 克隆项目仓库（含 adapter 权重和 demo 代码）
cd /root
git clone https://github.com/796joker/vqa-hallucination.git
cd vqa-hallucination

# 2. 下载基座模型（约 10-15 分钟）
# AutoDL 通常可以直接访问 HuggingFace
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir models/Qwen3-VL-8B-Instruct

# 如果 HuggingFace 被墙，使用镜像
# HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
#     --local-dir models/Qwen3-VL-8B-Instruct

# 3. 验证文件
ls models/Qwen3-VL-8B-Instruct/   # 应有 model*.safetensors, config.json 等
ls results/ablation/dpo_true_optimal/  # 应有 adapter_model.safetensors
ls demo/app.py demo/examples/        # 应有 app.py 和 10 张示例图
```

### 备选方案：从训练服务器 SCP

如果租卡平台下载慢，可以从 236 服务器直接传：

```bash
# 在 AutoDL 实例上执行（需 236 服务器能被访问）
scp -r research@115.190.215.236:/mnt/disk2/lijunlin/downloads/models/Qwen3-VL-8B-Instruct ./models/
scp -r research@115.190.215.236:/mnt/disk2/lijunlin/vqa-hallucination/results/ablation/dpo_true_optimal ./results/ablation/
scp -r research@115.190.215.236:/mnt/disk2/lijunlin/vqa-hallucination/demo ./
```

---

## 五、启动 Demo

### bf16 模式（≥40GB 显存）

```bash
cd /root/vqa-hallucination

python demo/app.py \
    --model_path models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_true_optimal \
    --port 6006 \
    --share
```

### 4-bit 量化模式（≥16GB 显存）

```bash
python demo/app.py \
    --model_path models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_true_optimal \
    --use_4bit \
    --port 6006 \
    --share
```

### 启动成功标志

```
[1/3] Loading processor from models/Qwen3-VL-8B-Instruct ...
[2/3] Loading base model ...
[3/3] Loading True Optimal model (adapter: results/ablation/dpo_true_optimal) ...

Models loaded. Starting Gradio on port 6006 ...
Running on local URL:  http://0.0.0.0:6006
Running on public URL: https://xxxxx.gradio.live  ← 用这个链接展示
```

> `--share` 会生成一个公网链接（72h 有效），课堂上可以直接打开展示。

---

## 六、AutoDL 特殊说明

### 端口映射

AutoDL 默认开放 6006 端口，所以用 `--port 6006`。如果 6006 被占用：

```bash
python demo/app.py --port 7860 --share
# 或在 AutoDL 控制台 → 自定义服务 → 添加端口映射 7860
```

### 加速下载

AutoDL 内置学术加速，通常可以直接从 HuggingFace 下载。如果不行：

```bash
# 使用 AutoDL 内置加速
source /etc/network_turbo

# 或设置镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 数据盘

模型 17GB，建议放在 `/root/autodl-tmp/`（数据盘，不计入系统盘用量）：

```bash
mkdir -p /root/autodl-tmp/models
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen3-VL-8B-Instruct

# 启动时指向数据盘
python demo/app.py \
    --model_path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_true_optimal \
    --port 6006 --share
```

---

## 七、Demo 功能说明（6 个 Tab）

| Tab | 功能 | 展示重点 |
|-----|------|---------|
| **对比演示** | Base vs True Optimal 并排回答 | 上传图片 → 选择问题 → 对比差异 |
| **幻觉检测 (POPE)** | 8 物体批量 yes/no 测试 | 展示 yes-bias 纠正效果 |
| **描述对比 (CHAIR)** | 图像描述生成对比 | 展示幻觉减少、描述更准确 |
| **实验指标** | 核心数据表 + 四大发现 | 无需 GPU，纯展示 |
| **消融实验** | 五维度消融结果 | 无需 GPU，纯展示 |
| **关于** | 项目信息 + 环境配置 | 无需 GPU，纯展示 |

### 展示建议

1. 先展示 Tab 4/5（实验指标），**不需要等模型加载**
2. 模型加载完后切到 Tab 1（对比演示），用示例图片演示
3. 展示 2-3 个成功案例 + 1 个失败案例
4. 切到 Tab 2（POPE 检测）展示 yes-bias 纠正

---

## 八、备份方案：录屏

如果展示当天租卡/网络有问题，提前录制备份视频：

```bash
# 在 236 服务器或 AutoDL 上运行 Demo 后
# 用 OBS 或系统录屏工具录制以下操作：
# 1. Tab 1: 上传 3 张不同图片，对比 Base vs Optimal
# 2. Tab 2: 运行一次 POPE 检测
# 3. Tab 3: 生成一次描述对比
# 4. Tab 4/5: 快速展示实验数据

# 视频建议 3-5 分钟，1080p
```

---

## 九、快速部署清单

```
□ 1. 租卡（A100-40G 或 RTX 4090）
□ 2. 选择 PyTorch 2.4+ / CUDA 12.x / Python 3.10+ 镜像
□ 3. pip install transformers peft accelerate gradio qwen-vl-utils
□ 4. git clone 项目仓库
□ 5. 下载 Qwen3-VL-8B-Instruct（~17GB, ~10min）
□ 6. python demo/app.py --model_path ... --adapter_path ... --port 6006 --share
□ 7. 等待 ~2 分钟模型加载
□ 8. 获得 gradio.live 公网链接 → 课堂展示
```

**预计从开机到可展示：15-20 分钟（含下载）。建议课前 30 分钟启动。**
