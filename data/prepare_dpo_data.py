"""
Convert openbmb/RLAIF-V-Dataset to LLaMA-Factory DPO format.

This is OPTIONAL -- the built-in `rlhf_v` dataset (5.7K samples) can be used directly.
Use this script only if you want the larger RLAIF-V dataset (83K samples).

Supports two modes:
  1. --input_dir: Load from local parquet files (offline, recommended)
  2. (default):   Download from HuggingFace (requires network access)

Output format (LLaMA-Factory ranking sharegpt):
  {"conversations": [{"from": "human", "value": "<image>\nQuestion"}],
   "chosen": {"from": "gpt", "value": "faithful answer"},
   "rejected": {"from": "gpt", "value": "hallucinated answer"},
   "images": ["path/to/image.jpg"]}
"""

import argparse
import glob
import json
import os
from io import BytesIO
from PIL import Image


def load_rlaifv(input_dir: str | None = None):
    """Load RLAIF-V dataset from local parquets or HuggingFace."""
    if input_dir:
        import pandas as pd
        parquets = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
        if not parquets:
            raise FileNotFoundError(f"No parquet files found in {input_dir}")
        print(f"Loading {len(parquets)} parquet files from {input_dir} ...")
        dfs = [pd.read_parquet(p) for p in parquets]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(df)} samples from local parquets")
        return df
    else:
        from datasets import load_dataset
        print("Downloading from HuggingFace openbmb/RLAIF-V-Dataset ...")
        ds = load_dataset("openbmb/RLAIF-V-Dataset", split="train")
        return ds


def save_image_from_parquet(image_data, save_path: str):
    """Extract and save image from parquet embedded format {bytes, path}."""
    if isinstance(image_data, dict) and "bytes" in image_data:
        img = Image.open(BytesIO(image_data["bytes"]))
    elif isinstance(image_data, Image.Image):
        img = image_data
    else:
        raise ValueError(f"Unknown image format: {type(image_data)}")
    img.convert("RGB").save(save_path, "JPEG", quality=85)


def convert_rlaifv(output_json: str, image_dir: str,
                   max_samples: int | None = None,
                   input_dir: str | None = None):
    data = load_rlaifv(input_dir)
    os.makedirs(image_dir, exist_ok=True)

    n = len(data) if hasattr(data, '__len__') else None
    if max_samples and n and max_samples < n:
        print(f"Limiting to {max_samples} of {n} samples")

    converted = []
    for i in range(len(data)):
        if max_samples and i >= max_samples:
            break

        if hasattr(data, 'iloc'):
            item = data.iloc[i]
        else:
            item = data[i]

        img_path = os.path.join(image_dir, f"rlaifv_{i:06d}.jpg")
        if not os.path.exists(img_path):
            save_image_from_parquet(item["image"], img_path)

        # LLaMA-Factory resolves images relative to dataset_dir (data/),
        # so strip "data/" prefix if image_dir starts with it
        img_ref = img_path
        if img_ref.startswith("data/"):
            img_ref = img_ref[5:]

        converted.append({
            "conversations": [
                {"from": "human", "value": f"<image>\n{item['question']}"}
            ],
            "chosen": {"from": "gpt", "value": item["chosen"]},
            "rejected": {"from": "gpt", "value": item["rejected"]},
            "images": [img_ref]
        })

        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1} / {max_samples or n} samples...")

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} DPO samples -> {output_json}")
    print(f"Images saved to {image_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RLAIF-V to LLaMA-Factory DPO format")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Local directory with RLAIF-V parquet files (offline mode)")
    parser.add_argument("--output", type=str, default="data/dpo_data/rlaifv_dpo.json")
    parser.add_argument("--image_dir", type=str, default="data/dpo_data/rlaifv_images",
                        help="Directory to save extracted images. Paths in output JSON "
                             "will be relative to data/ (LLaMA-Factory dataset_dir).")
    parser.add_argument("--max_samples", type=int, default=20000)
    args = parser.parse_args()

    convert_rlaifv(args.output, args.image_dir, args.max_samples, args.input_dir)
