#!/bin/bash
# ============================================================
# 全局路径配置 - 只需修改这里，所有脚本自动生效
# ============================================================

# 项目根目录 (自动检测)
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 基础存储目录 (大容量磁盘)
STORAGE_DIR="$(dirname "${PROJECT_DIR}")"

# 模型本地路径
export MODEL_PATH="${STORAGE_DIR}/downloads/models/Qwen3-VL-8B-Instruct"

# COCO 图像目录
export COCO_TRAIN_DIR="${STORAGE_DIR}/downloads/coco/train2017"
export COCO_VAL_DIR="${STORAGE_DIR}/downloads/coco/val2014"

# 避免写满系统盘：缓存和临时文件都放大盘
export HF_HOME="${STORAGE_DIR}/hf_cache"
export TMPDIR="${STORAGE_DIR}/tmp"
export HF_ENDPOINT=https://hf-api.gitee.com
mkdir -p "${HF_HOME}" "${TMPDIR}"
