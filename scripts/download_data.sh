#!/bin/bash
# Download all datasets for the project
# Run on A100 server

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "============================================"
echo "  数据下载与准备"
echo "  COCO train: ${COCO_TRAIN_DIR}"
echo "  COCO val:   ${COCO_VAL_DIR}"
echo "============================================"

# === Step 1: COCO images ===
echo ""
echo "[1/4] Downloading COCO images..."
mkdir -p "$(dirname "${COCO_TRAIN_DIR}")"
cd "$(dirname "${COCO_TRAIN_DIR}")"

if [ ! -d "${COCO_TRAIN_DIR}" ]; then
    echo "  Downloading COCO train2017 (~18GB)..."
    wget -c http://images.cocodataset.org/zips/train2017.zip
    unzip -q train2017.zip
    rm train2017.zip
else
    echo "  COCO train2017 already exists, skipping."
fi

if [ ! -d "${COCO_VAL_DIR}" ]; then
    echo "  Downloading COCO val2014 (~6GB)..."
    wget -c http://images.cocodataset.org/zips/val2014.zip
    unzip -q val2014.zip
    rm val2014.zip
else
    echo "  COCO val2014 already exists, skipping."
fi

# === Step 2: LLaVA-Instruct-150K ===
echo ""
echo "[2/4] Pre-caching LLaVA-Instruct-150K..."
python -c "
from datasets import load_dataset
ds = load_dataset('liuhaotian/LLaVA-Instruct-150K', split='train')
print(f'  LLaVA-Instruct-150K: {len(ds)} samples')
"

# === Step 3: POPE benchmark ===
echo ""
echo "[3/4] Pre-caching POPE benchmark..."
python -c "
from datasets import load_dataset
ds = load_dataset('lmms-lab/POPE')
for split in ds:
    print(f'  POPE {split}: {len(ds[split])} samples')
"

# === Step 4: Prepare formatted data ===
echo ""
echo "[4/4] Preparing formatted data..."
cd "${PROJECT_DIR}"
python data/prepare_sft_data.py \
    --coco_dir "${COCO_TRAIN_DIR}" \
    --output data/sft_data/llava_sft.json \
    --max_samples 50000

python data/prepare_pope.py \
    --coco_val_dir "${COCO_VAL_DIR}" \
    --output_dir data/pope_data

echo ""
echo "=== 数据准备完成 ==="
echo "SFT data: ${PROJECT_DIR}/data/sft_data/llava_sft.json"
echo "POPE data: ${PROJECT_DIR}/data/pope_data/"
