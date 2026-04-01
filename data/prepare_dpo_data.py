"""
Convert openbmb/RLAIF-V-Dataset to LLaMA-Factory DPO format.

This is OPTIONAL -- the built-in `rlhf_v` dataset (5.7K samples) can be used directly.
Use this script only if you want the larger RLAIF-V dataset (83K samples).

Output format (LLaMA-Factory ranking sharegpt):
  {"conversations": [{"from": "human", "value": "<image>\nQuestion"}],
   "chosen": {"from": "gpt", "value": "faithful answer"},
   "rejected": {"from": "gpt", "value": "hallucinated answer"},
   "images": ["path/to/image.jpg"]}
"""

import argparse
import json
import os
from datasets import load_dataset


def convert_rlaifv(output_json: str, image_dir: str, max_samples: int | None = None):
    ds = load_dataset("openbmb/RLAIF-V-Dataset", split="train")
    os.makedirs(image_dir, exist_ok=True)

    converted = []
    for i, item in enumerate(ds):
        if max_samples and i >= max_samples:
            break

        img_path = os.path.join(image_dir, f"rlaifv_{i:06d}.jpg")
        if not os.path.exists(img_path):
            item["image"].save(img_path)

        converted.append({
            "conversations": [
                {"from": "human", "value": f"<image>\n{item['question']}"}
            ],
            "chosen": {"from": "gpt", "value": item["chosen"]},
            "rejected": {"from": "gpt", "value": item["rejected"]},
            "images": [img_path]
        })

        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1} samples...")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} DPO samples -> {output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RLAIF-V to LLaMA-Factory DPO format")
    parser.add_argument("--output", type=str, default="data/dpo_data/rlaifv_dpo.json")
    parser.add_argument("--image_dir", type=str, default="data/dpo_data/images")
    parser.add_argument("--max_samples", type=int, default=20000)
    args = parser.parse_args()

    convert_rlaifv(args.output, args.image_dir, args.max_samples)
