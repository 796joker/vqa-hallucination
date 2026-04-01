"""
Convert LLaVA-Instruct-150K to LLaMA-Factory sharegpt multimodal format.

Input (LLaVA format):
  {"id": "000000215677", "image": "000000215677.jpg",
   "conversations": [{"from": "human", "value": "<image>\nWhat..."}, {"from": "gpt", "value": "..."}]}

Output (LLaMA-Factory sharegpt format):
  {"messages": [{"role": "user", "content": "<image>\nWhat..."}, {"role": "assistant", "content": "..."}],
   "images": ["train2017/000000215677.jpg"]}
"""

import argparse
import json
import os
from datasets import load_dataset


def convert_llava_to_llamafactory(coco_dir: str, output_path: str, max_samples: int | None = None):
    ds = load_dataset("liuhaotian/LLaVA-Instruct-150K", data_files="llava_instruct_150k.json", split="train")

    converted = []
    skipped = 0

    for i, item in enumerate(ds):
        if max_samples and len(converted) >= max_samples:
            break

        image_filename = item["image"]
        image_path = os.path.join(coco_dir, image_filename)

        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            skipped += 1
            continue

        messages = []
        for turn in item["conversations"]:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})

        converted.append({
            "messages": messages,
            "images": [image_path]
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(converted)} samples (skipped {skipped} missing images) -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LLaVA-Instruct-150K to LLaMA-Factory format")
    parser.add_argument("--coco_dir", type=str, required=True, help="Path to COCO train2017 images directory")
    parser.add_argument("--output", type=str, default="data/sft_data/llava_sft.json", help="Output JSON path")
    parser.add_argument("--max_samples", type=int, default=50000, help="Max samples to convert")
    args = parser.parse_args()

    convert_llava_to_llamafactory(args.coco_dir, args.output, args.max_samples)
