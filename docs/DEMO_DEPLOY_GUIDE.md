# Gradio Demo 部署指南（AutoDL / FunHPC）

> 课程展示用：在租卡平台上部署 Gradio Demo，展示 Base vs True Optimal 的幻觉缓解效果。
>
> **建议课前 30-60 分钟启动部署，预计 20-25 分钟完成。**

---

## 快速清单（TL;DR）

### AutoDL

```
□  1. 租卡 — A100-40GB (bf16) 或 RTX 4090 (4-bit)         ← 5 min
□  2. 镜像 — PyTorch 2.4+ / CUDA 12.x / Python 3.10+      ← 1 min
□  3. 加速 — source /etc/network_turbo (AutoDL)             ← 10 sec
□  4. 克隆 — git clone + git lfs pull                       ← 2 min
□  5. 依赖 — pip install -r requirements.txt                ← 3 min
□  6. 模型 — hf download 基座模型                           ← 10 min
□  7. 图片 — python demo/download_examples.py               ← 30 sec
□  8. 启动 — python demo/app.py --share                     ← 2 min (模型加载)
□  9. 链接 — 获得 https://xxxxx.gradio.live                 ← 即时
□ 10. 试跑 — 按 Demo 脚本过一遍                            ← 5 min
```

### FunHPC

```
□  1. 租卡 — A100-40GB (bf16) 或 RTX 4090 (4-bit)         ← 5 min
□  2. 镜像 — PyTorch 2.4+ / CUDA 12.x / Python 3.10+      ← 1 min
□  3. 镜像源 — pip config + HF_ENDPOINT 配置               ← 1 min
□  4. 克隆 — git clone（可能需 gitclone.com 镜像）          ← 2 min
□  5. 依赖 — pip install -r requirements.txt                ← 3 min
□  6. 模型 — HF_ENDPOINT=hf-mirror.com hf download         ← 10 min
□  7. 图片 — python demo/download_examples.py               ← 30 sec
□  8. 启动 — python demo/app.py --share                     ← 2 min (模型加载)
□  9. 链接 — 获得 https://xxxxx.gradio.live                 ← 即时
□ 10. 试跑 — 按 Demo 脚本过一遍                            ← 5 min
```

---

## 一、GPU 选型

### 显存需求

| 模式 | 加载内容 | 显存需求 | 推荐卡型 |
|------|---------|---------|---------|
| **bf16（推荐）** | Base 模型 ~16GB + Fine-tuned ~16GB | **~32GB** | A100-40GB, A6000-48GB |
| **4-bit 量化** | Base ~5GB + Fine-tuned ~5GB | **~12GB** | RTX 4090-24GB, A5000-24GB |

### AutoDL 推荐

| 优先级 | 卡型 | 显存 | 价格参考 | 说明 |
|--------|------|------|---------|------|
| 1 | **A100-SXM4-40GB** | 40GB | ~3 元/h | bf16 全精度，推理效果最好 |
| 2 | **A6000-48GB** | 48GB | ~2.5 元/h | 性价比高 |
| 3 | **RTX 4090-24GB** | 24GB | ~1.5 元/h | 需 `--use_4bit`，最便宜 |

> 展示约 15-30 分钟，含准备总费用约 1-3 元。

### FunHPC 推荐

选择 **PyTorch** 框架 → CUDA 12.x → 显存 ≥24GB → Python 3.10+。与 AutoDL 卡型基本一致，优先选 A100-40GB（bf16 全精度）或 RTX 4090（需 `--use_4bit`）。FunHPC 的价格和可用卡型请查阅平台首页。

---

## 二、镜像选择

### 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10 - 3.12 | 3.12 最佳（与训练环境一致） |
| CUDA | 12.1+ | 12.4 最佳 |
| PyTorch | 2.4.0+ | 预装即可 |

### AutoDL 镜像

1. **首选**: `PyTorch 2.4.0 + Python 3.12 + CUDA 12.4`
2. **备选**: `PyTorch 2.3.0 + Python 3.10 + CUDA 12.1`
3. 如无上述，选 `PyTorch 2.x + CUDA 12.x` 最新版

### FunHPC 镜像

选择 **PyTorch** 框架，CUDA 12.x，Python 3.10+。FunHPC 的镜像选择界面可能不同于 AutoDL，核心要求是确保 PyTorch ≥2.4 + CUDA 12.x 可用。如有多个版本可选，优先选最新的。

---

## 三、环境配置

### AutoDL

AutoDL 镜像自带 Miniconda（`/root/miniconda3/`），选择 PyTorch 镜像后 `base` 环境已预装 PyTorch + CUDA，**直接在 base 环境中 pip install 即可，无需创建虚拟环境**。pip 已预配置阿里云镜像源，下载速度正常（高峰期如果限速，可在 AutoDL 控制台「软件源」切换清华/中科大源）。

登录实例后执行（**整段复制粘贴**）：

```bash
# ===== AutoDL 专用：启用学术资源加速（GitHub/HuggingFace）=====
source /etc/network_turbo 2>/dev/null

# ===== 1. 克隆仓库 =====
cd /root
git lfs install
git clone https://github.com/796joker/vqa-hallucination.git
cd vqa-hallucination

# ===== 2. 安装额外依赖（约 2-3 分钟）=====
# AutoDL 镜像已有 torch，只需装剩余包
pip install -r requirements.txt

# 如果用 4-bit 量化（RTX 4090 等 <32GB 显存的卡），额外安装：
# pip install bitsandbytes

# ===== 3. 验证安装 =====
python -c "
import torch, transformers, peft, gradio, accelerate
print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'gradio: {gradio.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f'VRAM: {vram:.1f} GB')
if vram < 32: print('NOTE: VRAM < 32GB, use --use_4bit flag')
"
```

> **注**：`source /etc/network_turbo` 是加速 GitHub/HuggingFace 等学术资源的代理，与 pip 镜像源无关。pip 下载走的是 AutoDL 预配置的国内镜像，不需要代理。

### FunHPC

FunHPC 平台的 PyTorch 镜像通常预装 PyTorch + CUDA，但 **pip 镜像源可能未预配置**，需要手动设置以保证国内下载速度。

登录实例后执行（**整段复制粘贴**）：

```bash
# ===== 1. 配置 pip 国内镜像源（重要！）=====
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# ===== 2. 配置 HuggingFace 镜像（FunHPC 无 network_turbo）=====
export HF_ENDPOINT=https://hf-mirror.com

# 建议写入 ~/.bashrc 避免每次登录重设
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# ===== 3. 配置 Git 代理（如 GitHub 克隆慢）=====
# 方法 A: 使用 gitclone 镜像
# git clone https://gitclone.com/github.com/796joker/vqa-hallucination.git
# 方法 B: 直接克隆（如果平台网络支持）
cd /root
git lfs install
git clone https://github.com/796joker/vqa-hallucination.git
cd vqa-hallucination

# ===== 4. 安装依赖 =====
pip install -r requirements.txt
# 4-bit 量化（显存 <32GB）：
# pip install bitsandbytes

# ===== 5. 验证安装 =====
python -c "
import torch, transformers, peft, gradio, accelerate
print(f'torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'gradio: {gradio.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f'VRAM: {vram:.1f} GB')
if vram < 32: print('NOTE: VRAM < 32GB, use --use_4bit flag')
"
```

> **注**：FunHPC 的具体网络环境、存储路径和端口映射方式可能随版本更新变化，请查阅 [FunHPC 官方文档](https://www.funhpc.com/#/documentation/introduction) 确认最新信息。

---

## 四、下载基座模型

模型约 17GB。

### AutoDL

建议放在数据盘 `/root/autodl-tmp/` 以节省系统盘空间（系统盘通常仅 30GB）：

```bash
mkdir -p /root/autodl-tmp/models
hf download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen3-VL-8B-Instruct
```

如果下载慢（<1MB/s），先执行 `source /etc/network_turbo`，或使用镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir /root/autodl-tmp/models/Qwen3-VL-8B-Instruct
```

### FunHPC

FunHPC 没有 `network_turbo` 加速，**必须使用 HF 镜像下载**。存储路径请根据平台实际磁盘结构选择（查看平台文档了解数据盘挂载点）：

```bash
# 设置 HF 镜像（如果未在 ~/.bashrc 中配置）
export HF_ENDPOINT=https://hf-mirror.com

# 下载模型（路径根据平台调整）
mkdir -p /root/models
hf download Qwen/Qwen3-VL-8B-Instruct \
    --local-dir /root/models/Qwen3-VL-8B-Instruct
```

> **注**：FunHPC 部分实例可能有独立数据盘（如 `/data/` 或 `/mnt/`），建议将模型放在数据盘上。具体路径请查阅平台文档或执行 `df -h` 查看磁盘挂载情况。

### 下载示例图片（通用）

```bash
cd /root/vqa-hallucination
python demo/download_examples.py

# 如果 COCO CDN 被墙，使用镜像：
# python demo/download_examples.py --mirror
```

### 验证文件完整性（通用）

```bash
# 基座模型（应有 model*.safetensors, config.json 等）
# AutoDL:
ls /root/autodl-tmp/models/Qwen3-VL-8B-Instruct/*.safetensors | wc -l  # 应为 5
# FunHPC:
# ls /root/models/Qwen3-VL-8B-Instruct/*.safetensors | wc -l  # 应为 5

# Adapter 权重（随仓库克隆，约 83MB）
ls -lh results/ablation/dpo_true_optimal/adapter_model.safetensors

# 示例图片
ls demo/examples/*.jpg | wc -l  # 应为 10
```

---

## 五、启动 Demo

### bf16 模式（≥40GB 显存）

```bash
cd /root/vqa-hallucination

# AutoDL（模型在数据盘）
python demo/app.py \
    --model_path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_true_optimal \
    --port 6006 \
    --share

# FunHPC（模型路径根据实际下载位置调整）
# python demo/app.py \
#     --model_path /root/models/Qwen3-VL-8B-Instruct \
#     --adapter_path results/ablation/dpo_true_optimal \
#     --port 6006 \
#     --share
```

### 4-bit 量化模式（≥16GB 显存，如 RTX 4090）

```bash
# AutoDL
python demo/app.py \
    --model_path /root/autodl-tmp/models/Qwen3-VL-8B-Instruct \
    --adapter_path results/ablation/dpo_true_optimal \
    --use_4bit \
    --port 6006 \
    --share

# FunHPC
# python demo/app.py \
#     --model_path /root/models/Qwen3-VL-8B-Instruct \
#     --adapter_path results/ablation/dpo_true_optimal \
#     --use_4bit \
#     --port 6006 \
#     --share
```

### 启动成功标志

```
============================================================
VQA Hallucination Mitigation Demo
============================================================
  Base model:  /root/autodl-tmp/models/Qwen3-VL-8B-Instruct
  Adapter:     /root/vqa-hallucination/results/ablation/dpo_true_optimal
  4-bit:       False
  Port:        6006
  Examples:    10 images
  GPU:         NVIDIA A100-SXM4-40GB
  VRAM:        40.0 GB
============================================================

[1/3] Loading processor ...
[2/3] Loading base model ...
[3/3] Loading True Optimal model ...

Models loaded. Starting Gradio on port 6006 ...
Running on local URL:  http://0.0.0.0:6006
Running on public URL: https://xxxxx.gradio.live  ← 课堂展示用这个链接
```

> `--share` 生成的公网链接 72 小时有效，课堂上直接打开即可。

---

## 六、5 分钟 Demo 脚本

课程要求 5 分钟 live demo，展示 2-3 个成功案例 + 边界案例。按以下脚本进行：

| 时间 | 操作 | Tab | 要点 |
|------|------|-----|------|
| 0:00-0:30 | 开场介绍 | — | "我们对 Qwen3-VL-8B 进行了 SFT+DPO 后训练来缓解幻觉，下面对比基座模型和优化后模型的差异。" |
| **0:30-1:30** | **成功案例 1：图像描述** | 对比演示 | 点击示例图片，选"详细描述"。指出：Base 生成更长文本但含幻觉物体，True Optimal 更简洁准确。 |
| **1:30-2:30** | **成功案例 2：物体判别** | 对比演示 | 同一图片，输入自定义问题 "Is there a [不存在的物体] in this image? Answer yes or no."。展示 Base 回答 yes（幻觉），True Optimal 回答 no（正确）。 |
| **2:30-3:30** | **POPE 批量测试** | 幻觉检测 | 点击示例图片，运行检测。指出 Yes-Ratio 差异——Base 偏高（容易说 yes），True Optimal 更平衡。 |
| **3:30-4:15** | **边界案例** | 对比演示 | 换一张复杂场景图片，问共现物体。展示两个模型都失败（如图中有 truck，问 car，两者都说 yes）。说明："DPO 降低了 39.6% 的幻觉但无法完全消除。" |
| **4:15-4:45** | 指标总结 | 实验指标 | 快速浏览数据表。强调：CHAIR_i 33.31% → 20.12%，MME 能力保持 99.1%，训练仅 1 小时。 |
| **4:45-5:00** | 结语 | — | "1 小时训练、~2.5 元成本，幻觉降低 39.6% 同时保持 99.1% 通用能力。" |

### 展示技巧

- **等待生成时持续讲解**：模型推理需要 5-15 秒，利用这段时间解释背景
- **先准备好问题再上传图片**：节省等待时间
- **边界案例不必刻意隐藏**：课程评分标准要求展示模型局限性，诚实展示加分
- **Tab 4/5 不需要 GPU**：如果模型加载慢，可以先展示实验指标页面

### 推荐的边界案例问题

| 场景 | 问题示例 | 预期行为 |
|------|---------|---------|
| 图中有卡车 | "Is there a car?" | 两模型可能都说 yes（共现混淆） |
| 远处小物体 | "Is there a clock?" | 可能漏检 |
| 复杂餐桌场景 | "详细描述" | 都可能幻觉餐具 |

---

## 七、备份录屏

课程要求有备份方案。部署成功后立即录制一段备份视频：

### 录制步骤

1. 确认 Demo 正常运行（至少跑通一次完整流程）
2. 打开录屏工具（macOS: QuickTime; Windows: OBS / 系统录屏）
3. 设置分辨率 1920×1080，帧率 30fps
4. 按照上述 5 分钟脚本完整录制，同时口述讲解
5. 导出为 MP4（控制在 50MB 以内）
6. 上传到可靠位置（网盘、GitHub Release 等）

### 录制技巧

- 鼠标移动缓慢，让观众跟上
- 每次模型输出后停顿 2-3 秒再继续
- 录两遍，选效果好的一遍
- 如果网络不稳定，建议在 **出发前一天** 提前录好

---

## 八、平台特殊说明

### AutoDL

#### 端口映射

AutoDL 默认开放 6006 端口。如果被占用：

```bash
# 方法 1: 换端口
python demo/app.py --port 7860 --share

# 方法 2: 释放端口
kill -9 $(lsof -t -i:6006) 2>/dev/null
```

#### 数据盘 vs 系统盘

| 路径 | 说明 | 用途 |
|------|------|------|
| `/root/` | 系统盘（通常 30GB） | 代码、依赖 |
| `/root/autodl-tmp/` | 数据盘（通常 50-100GB） | 基座模型（17GB） |

#### 网络加速

```bash
# AutoDL 内置学术加速（一定要先执行）
source /etc/network_turbo

# 如果 HuggingFace 仍然慢
export HF_ENDPOINT=https://hf-mirror.com
```

### FunHPC

#### 端口映射

FunHPC 的端口映射规则可能与 AutoDL 不同。建议：

1. **优先使用 `--share`**：Gradio 的 `--share` 参数会生成公网链接（`https://xxxxx.gradio.live`），无需关心端口映射，适用于所有平台
2. 如果需要本地端口访问，查阅 FunHPC 文档了解端口开放规则
3. 默认使用 6006 端口，如被占用换 7860

```bash
# 通用方式：--share 生成公网链接
python demo/app.py --port 6006 --share
```

#### 存储路径

FunHPC 实例的磁盘结构因机型和套餐不同而异。登录后执行以下命令了解：

```bash
df -h          # 查看磁盘挂载和容量
ls /root/      # 查看主目录
ls /data/ 2>/dev/null    # 部分实例有 /data/ 数据盘
ls /mnt/ 2>/dev/null     # 部分实例有 /mnt/ 挂载
```

将基座模型（17GB）放在容量最大的分区上。

#### 网络与下载加速

FunHPC 没有类似 AutoDL 的 `network_turbo` 命令，需要手动配置镜像：

```bash
# 1. pip 镜像（必须配置，否则 pip install 很慢）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 2. HuggingFace 镜像（必须配置，否则模型下载不了）
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# 3. GitHub 克隆加速（如果 git clone 慢）
# 方法 A: gitclone.com 镜像
git clone https://gitclone.com/github.com/796joker/vqa-hallucination.git
# 方法 B: ghproxy 镜像
# git clone https://ghproxy.com/https://github.com/796joker/vqa-hallucination.git
```

#### 环境管理

FunHPC 的 PyTorch 镜像通常也预装 Miniconda/Anaconda。如果 `base` 环境已有 PyTorch，直接 pip install 即可。如果没有预装：

```bash
# 检查 PyTorch 是否可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 如果没有预装，安装 PyTorch（需匹配 CUDA 版本）
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## 九、故障排查

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| GPU OOM | `CUDA out of memory` | 添加 `--use_4bit` 参数 |
| 模型下载失败 | `Connection refused` / 超时 | AutoDL: `source /etc/network_turbo`；FunHPC: `export HF_ENDPOINT=https://hf-mirror.com` |
| pip 下载慢 | 安装依赖超过 10 分钟 | AutoDL: 已有镜像，检查软件源设置；FunHPC: `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple` |
| GitHub 克隆慢 | git clone 超时 | AutoDL: `source /etc/network_turbo`；FunHPC: 用 `https://gitclone.com/github.com/...` 镜像 |
| gradio.live 链接未生成 | 无 public URL | 确认 `--share` 参数；或检查平台端口映射设置 |
| 推理很慢 | >30 秒/问题 | bf16 on A100 正常 ~10s；4-bit 更快 ~5s |
| 示例图片缺失 | UI 提示"未检测到示例图片" | `python demo/download_examples.py` |
| 端口被占用 | `Address already in use` | `kill -9 $(lsof -t -i:6006)` 或换端口 |
| LFS 文件是占位符 | adapter 文件 <1KB | `git lfs pull` |
| Adapter 加载报错 | `PeftModel` 异常 | 检查 peft 版本 == 0.18.1 |
| Gradio 版本不兼容 | UI 崩溃 / API 变更 | `pip install gradio==5.50.0` |

---

## 十、一键部署脚本

### AutoDL 版

将以下内容保存为 `setup_autodl.sh`，在实例上执行 `bash setup_autodl.sh` 即可：

```bash
#!/bin/bash
set -e

echo "===== VQA Demo Setup (AutoDL) ====="

# 网络加速 (AutoDL)
source /etc/network_turbo 2>/dev/null || true

# 克隆仓库
cd /root
if [ ! -d "vqa-hallucination" ]; then
    git lfs install
    git clone https://github.com/796joker/vqa-hallucination.git
fi
cd vqa-hallucination

# 安装依赖
pip install -r requirements.txt

# 下载基座模型
MODEL_DIR="/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p /root/autodl-tmp/models
    hf download Qwen/Qwen3-VL-8B-Instruct \
        --local-dir "$MODEL_DIR"
fi

# 下载示例图片
python demo/download_examples.py

echo ""
echo "===== Setup Complete ====="
echo "Launch command:"
echo "  python demo/app.py --model_path $MODEL_DIR --adapter_path results/ablation/dpo_true_optimal --port 6006 --share"
echo ""
echo "For 4-bit (RTX 4090):"
echo "  pip install bitsandbytes"
echo "  python demo/app.py --model_path $MODEL_DIR --adapter_path results/ablation/dpo_true_optimal --use_4bit --port 6006 --share"
```

### FunHPC 版

将以下内容保存为 `setup_funhpc.sh`，在实例上执行 `bash setup_funhpc.sh` 即可：

```bash
#!/bin/bash
set -e

echo "===== VQA Demo Setup (FunHPC) ====="

# 配置 pip 镜像
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 配置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
grep -q 'HF_ENDPOINT' ~/.bashrc 2>/dev/null || \
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

# 克隆仓库（尝试镜像加速）
cd /root
if [ ! -d "vqa-hallucination" ]; then
    git lfs install
    git clone https://github.com/796joker/vqa-hallucination.git || \
    git clone https://gitclone.com/github.com/796joker/vqa-hallucination.git
fi
cd vqa-hallucination

# 安装依赖
pip install -r requirements.txt

# 下载基座模型（使用 HF 镜像）
MODEL_DIR="/root/models/Qwen3-VL-8B-Instruct"
if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p /root/models
    hf download Qwen/Qwen3-VL-8B-Instruct \
        --local-dir "$MODEL_DIR"
fi

# 下载示例图片
python demo/download_examples.py

echo ""
echo "===== Setup Complete ====="
echo "Launch command:"
echo "  python demo/app.py --model_path $MODEL_DIR --adapter_path results/ablation/dpo_true_optimal --port 6006 --share"
echo ""
echo "For 4-bit (RTX 4090):"
echo "  pip install bitsandbytes"
echo "  python demo/app.py --model_path $MODEL_DIR --adapter_path results/ablation/dpo_true_optimal --use_4bit --port 6006 --share"
echo ""
echo "NOTE: Model path is $MODEL_DIR. Adjust if your data disk is at a different location."
```

---

## 附：Demo 功能一览（6 个 Tab）

| Tab | 功能 | 是否需要 GPU |
|-----|------|:----------:|
| **对比演示** | Base vs True Optimal 并排回答 + 示例图片一键点击 | 是 |
| **幻觉检测 (POPE)** | 8 物体批量 yes/no 测试，对比 yes-bias | 是 |
| **描述对比 (CHAIR)** | 图像描述生成对比 | 是 |
| **实验指标** | 核心数据表 + 四大发现 | 否 |
| **消融实验** | 五维度消融结果 | 否 |
| **关于** | 项目信息 + 环境配置 | 否 |
