"""
Prepare POPE (Polling-based Object Probing Evaluation) benchmark.

POPE tests object hallucination with yes/no questions like:
  "Is there a [object] in the image?"

Three splits by negative sampling strategy:
  - random: random non-existent objects
  - popular: frequently mentioned but absent objects
  - adversarial: co-occurring but absent objects (hardest)
"""

import argparse
import json
import os
from datasets import load_dataset


def prepare_pope(coco_val_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("lmms-lab/POPE")

    for split_name in ds:
        split_data = ds[split_name]
        items = []
        missing_images = 0

        for row in split_data:
            # POPE uses COCO val2014 images
            image_file = row.get("image_source", row.get("image", ""))
            if not image_file.endswith(".jpg"):
                image_file = f"COCO_val2014_{image_file:012d}.jpg" if isinstance(image_file, int) else f"{image_file}.jpg"

            image_path = os.path.join(coco_val_dir, image_file)

            items.append({
                "question_id": row.get("question_id", len(items)),
                "image": image_path,
                "image_file": image_file,
                "question": row["question"],
                "answer": row["answer"].strip().lower(),
                "category": split_name
            })

        out_path = os.path.join(output_dir, f"pope_{split_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

        print(f"POPE {split_name}: {len(items)} questions -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare POPE benchmark")
    parser.add_argument("--coco_val_dir", type=str, required=True, help="Path to COCO val2014 images")
    parser.add_argument("--output_dir", type=str, default="data/pope_data")
    args = parser.parse_args()

    prepare_pope(args.coco_val_dir, args.output_dir)
