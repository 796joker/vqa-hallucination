# 下载指南：模型与数据集

## 总览

| 资源 | HuggingFace ID | 类型 | 大小 | 用途 |
|------|---------------|------|------|------|
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | 模型 | ~16GB | 基座模型 |
| LLaVA-Instruct-150K | `liuhaotian/LLaVA-Instruct-150K` | 数据集 | ~200MB | SFT 训练数据 (JSON) |
| COCO train2017 | cocodataset.org | 图像 | ~18GB | SFT 图像 (配合 LLaVA) |
| COCO val2014 | cocodataset.org | 图像 | ~6GB | POPE 评估图像 |
| RLHF-V | `llamafactory/RLHF-V` | 数据集 | ~1GB | DPO 训练 (LLaMA-Factory 自动下载) |
| POPE | `lmms-lab/POPE` | 数据集 | ~5MB | 幻觉评估基准 |

---

## 目录结构

所有下载统一存放到 `~/code/downloads/`：

```
~/code/downloads/
├── models/
│   └── Qwen3-VL-8B-Instruct/       # 基座模型
├── datasets/
│   ├── LLaVA-Instruct-150K/        # SFT 对话数据 (JSON)
│   ├── POPE/                        # 评估数据
│   └── RLHF-V/                     # DPO 偏好数据 (可选手动下载)
└── coco/
    ├── train2017/                   # ~118K 张图像
    └── val2014/                     # ~40K 张图像
```

请先创建目录：

```bash
mkdir -p ~/code/downloads/{models,datasets,coco}
```

---

## 1. 模型下载

### Qwen3-VL-8B-Instruct (~16GB)

```bash
cd ~/code/downloads/models
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir Qwen3-VL-8B-Instruct
```

验证：
```bash
ls ~/code/downloads/models/Qwen3-VL-8B-Instruct/
# 应该看到: config.json, model-*.safetensors, tokenizer.json, preprocessor_config.json 等
```

---

## 2. SFT 数据下载

### LLaVA-Instruct-150K (JSON, ~200MB)

```bash
cd ~/code/downloads/datasets
hf download liuhaotian/LLaVA-Instruct-150K --repo-type dataset --local-dir LLaVA-Instruct-150K
```

验证：
```bash
ls ~/code/downloads/datasets/LLaVA-Instruct-150K/
# 应该看到: llava_instruct_150k.json 等文件
```

### COCO train2017 图像 (~18GB)

LLaVA-Instruct-150K 的图像来自 COCO，需要单独下载。**不在 HuggingFace 上**，用 wget：

```bash
cd ~/code/downloads/coco
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip -q train2017.zip && rm train2017.zip
```

如果下载慢，可以先在本地下载再传到服务器：
```bash
scp train2017.zip user@server:~/code/downloads/coco/
```

---

## 3. DPO 数据下载

### 方案 A：使用 LLaMA-Factory 内置 rlhf_v (推荐)

DPO 配置中 `dataset: rlhf_v` 会让 LLaMA-Factory **自动下载** `llamafactory/RLHF-V` (5.7K 样本)。

**无需任何手动操作**，首次训练时自动下载。

### 方案 B：手动下载更大的 RLAIF-V (可选，83K 样本)

```bash
cd ~/code/downloads/datasets
hf download openbmb/RLAIF-V-Dataset --repo-type dataset --local-dir RLAIF-V
```

下载后运行转换脚本，并将 DPO 配置中 `dataset` 从 `rlhf_v` 改为 `rlaifv_dpo`，详见配置修改章节。

---

## 4. 评估数据下载

### POPE 评估基准 (~5MB)

```bash
cd ~/code/downloads/datasets
hf download lmms-lab/POPE --repo-type dataset --local-dir POPE
```

### COCO val2014 图像 (~6GB, POPE 评估需要)

```bash
cd ~/code/downloads/coco
wget -c http://images.cocodataset.org/zips/val2014.zip
unzip -q val2014.zip && rm val2014.zip
```

---

## 5. 下载完成后：配置说明

**所有路径已预配置好，下载到指定目录后无需修改任何文件。**

项目使用相对路径，结构如下：

```
~/code/
├── downloads/          # 下载资源 (按上述步骤下载)
└── vqa-hallucination/  # 项目代码
```

- `configs/*.yaml` 中 `model_name_or_path: ../downloads/models/Qwen3-VL-8B-Instruct`
- `scripts/config.sh` 中 COCO 路径指向 `../downloads/coco/...`
- 所有 shell 脚本自动读取 `config.sh`

只要 `downloads/` 和 `vqa-hallucination/` 在同一个父目录下，无需改任何配置。

---

## 6. 完整下载命令汇总

依次在服务器上执行：

```bash
# 0. 创建目录
mkdir -p ~/code/downloads/{models,datasets,coco}

# 1. 模型 (~16GB)
cd ~/code/downloads/models
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir Qwen3-VL-8B-Instruct

# 2. SFT 数据 (~200MB)
cd ~/code/downloads/datasets
hf download liuhaotian/LLaVA-Instruct-150K --repo-type dataset --local-dir LLaVA-Instruct-150K

# 3. POPE 评估 (~5MB)
hf download lmms-lab/POPE --repo-type dataset --local-dir POPE

# 4. COCO train2017 (~18GB)
cd ~/code/downloads/coco
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip -q train2017.zip && rm train2017.zip

# 5. COCO val2014 (~6GB)
wget -c http://images.cocodataset.org/zips/val2014.zip
unzip -q val2014.zip && rm val2014.zip
```

DPO 数据 (RLHF-V) 无需手动下载，训练时自动获取。

---

## 常见问题

### Q: HuggingFace 下载太慢怎么办？
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 然后正常执行 hf download 命令
```

### Q: 下载中断了怎么办？
`hf download` 支持断点续传，重新执行相同命令即可。

### Q: COCO 图像下载太慢？
1. 本地先下载，再 scp/rsync 传到服务器
2. 搜索 COCO 数据集国内镜像

### Q: 提示需要登录？
```bash
hf auth login
# 输入你的 HuggingFace access token
```

### Q: 磁盘空间需要多少？

| 资源 | 大小 |
|------|------|
| 模型 | ~16GB |
| COCO train2017 | ~18GB |
| COCO val2014 | ~6GB |
| LLaVA JSON | ~200MB |
| POPE | ~5MB |
| RLHF-V (自动) | ~1GB |
| **合计** | **~41GB** |

加上训练 checkpoint (~5GB/个)，建议预留 **80-100GB** 磁盘空间。
