"""
Download and prepare MME benchmark data from HuggingFace.
Run this LOCALLY (not on the server, which can't access HuggingFace).

Output structure:
    data/mme/
    ├── existence/
    │   ├── 0001.jpg
    │   ├── 0001.txt
    │   └── ...
    ├── count/
    │   └── ...
    └── questions.json

Then scp data/mme/ to the server:
    scp -r data/mme research@115.190.215.236:/mnt/disk2/lijunlin/vqa-hallucination/data/

Usage:
    python scripts/prepare_mme.py --output_dir data/mme
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/mme")
    args = parser.parse_args()

    print("Loading MME dataset from HuggingFace (lmms-lab/MME)...")
    print("This may take a while for the first download (~864MB)")

    from datasets import load_dataset
    dataset = load_dataset("lmms-lab/MME", split="test")

    print(f"Loaded {len(dataset)} items")

    os.makedirs(args.output_dir, exist_ok=True)

    questions = []
    categories = set()
    img_counters = {}

    for idx, item in enumerate(dataset):
        category = item["category"]
        question = item["question"]
        answer = item["answer"]
        image = item["image"]  # PIL Image

        categories.add(category)

        # Create category directory
        cat_dir = os.path.join(args.output_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        # Generate image filename based on question_id or index
        q_id = item.get("question_id", str(idx))

        # MME pairs: same image has 2 questions. Use question_id to group.
        # The image filename should be shared between paired questions.
        # Extract base image id (remove trailing _0, _1 if present)
        if "_" in str(q_id):
            parts = str(q_id).rsplit("_", 1)
            if parts[-1].isdigit():
                img_id = parts[0]
                q_idx = int(parts[-1])
            else:
                img_id = str(q_id)
                q_idx = img_counters.get(f"{category}_{img_id}", 0)
        else:
            img_id = str(q_id)
            q_idx = img_counters.get(f"{category}_{img_id}", 0)

        counter_key = f"{category}_{img_id}"
        if counter_key not in img_counters:
            img_counters[counter_key] = 0

        img_filename = f"{img_id}.jpg"
        txt_filename = f"{img_id}.txt"
        img_path = os.path.join(cat_dir, img_filename)
        txt_path = os.path.join(cat_dir, txt_filename)

        # Save image (only once per image)
        if not os.path.exists(img_path):
            image.save(img_path)

        # Append question to txt file
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"{question}\t{answer}\n")

        img_counters[counter_key] += 1

        # Build questions.json entry
        split = "perception" if category in [
            "existence", "count", "position", "color", "posters",
            "celebrity", "scene", "landmark", "artwork", "OCR"
        ] else "cognition"

        questions.append({
            "question_id": f"{category}_{img_id}_{q_idx}",
            "image": os.path.join(cat_dir, img_filename),
            "image_file": img_filename,
            "question": question,
            "answer": answer,
            "category": category,
            "split": split,
        })

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)}")

    # Save questions.json
    json_path = os.path.join(args.output_dir, "questions.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(questions)} questions across {len(categories)} categories")
    print(f"Categories: {sorted(categories)}")
    print(f"Output: {args.output_dir}")
    print(f"\nNext step: scp to server:")
    print(f"  scp -r {args.output_dir} research@115.190.215.236:/mnt/disk2/lijunlin/vqa-hallucination/data/")


if __name__ == "__main__":
    main()
