"""
Generate model answers on MME benchmark.
MME: 14 subtasks (10 perception + 4 cognition), all yes/no questions.
Each image has paired questions (one yes, one no ground truth).

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval/generate_mme_answers.py \
        --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
        --adapter_path results/ablation/dpo_true_optimal \
        --mme_dir data/mme \
        --output_file results/eval/dpo_true_optimal/mme_answers.json
"""

import argparse
import json
import os
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


PERCEPTION_TASKS = [
    "existence", "count", "position", "color", "posters",
    "celebrity", "scene", "landmark", "artwork", "OCR"
]
COGNITION_TASKS = [
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning"
]
ALL_TASKS = PERCEPTION_TASKS + COGNITION_TASKS


def load_model(base_model_path: str, adapter_path: str | None = None):
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, processor


def generate_answer(model, processor, image_path: str, question: str) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    answer = processor.decode(generated, skip_special_tokens=True)
    # Clean think tags (same as POPE/CHAIR)
    answer = re.sub(r"</?think>", "", answer).strip()
    return answer


def load_mme_questions(mme_dir: str) -> list[dict]:
    """Load MME questions from the standard directory structure.

    Expected structure:
        mme_dir/
        ├── existence/
        │   ├── 0001.jpg
        │   ├── 0001.txt  (2 lines: question1\\tanswer1\\nquestion2\\tanswer2)
        │   └── ...
        ├── count/
        │   └── ...
        └── ...

    Or from preprocessed questions.json.
    """
    json_path = os.path.join(mme_dir, "questions.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Load from official directory structure
    questions = []
    for task in ALL_TASKS:
        task_dir = os.path.join(mme_dir, task)
        if not os.path.isdir(task_dir):
            print(f"Warning: task dir not found: {task_dir}")
            continue

        # Find all image files
        image_files = sorted([
            f for f in os.listdir(task_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        for img_file in image_files:
            img_name = os.path.splitext(img_file)[0]
            txt_file = os.path.join(task_dir, f"{img_name}.txt")
            if not os.path.exists(txt_file):
                continue

            with open(txt_file, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        q_text, gt_answer = parts[0], parts[1]
                    else:
                        continue

                    split = "perception" if task in PERCEPTION_TASKS else "cognition"
                    questions.append({
                        "question_id": f"{task}_{img_name}_{line_idx}",
                        "image": os.path.join(task_dir, img_file),
                        "image_file": img_file,
                        "question": q_text,
                        "answer": gt_answer.strip(),
                        "category": task,
                        "split": split,
                    })

    print(f"Loaded {len(questions)} MME questions from {len(set(q['category'] for q in questions))} tasks")
    return questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../downloads/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--mme_dir", type=str, default="data/mme")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated list of categories to evaluate (default: all)")
    args = parser.parse_args()

    questions = load_mme_questions(args.mme_dir)
    if not questions:
        print("No questions loaded!")
        return

    if args.categories:
        cats = [c.strip() for c in args.categories.split(",")]
        questions = [q for q in questions if q["category"] in cats]
        print(f"Filtered to {len(questions)} questions for categories: {cats}")

    # Skip if output already exists
    if os.path.exists(args.output_file):
        print(f"Output already exists: {args.output_file}")
        return

    model, processor = load_model(args.model_path, args.adapter_path)

    results = []
    for item in tqdm(questions, desc="MME evaluation"):
        if not os.path.exists(item["image"]):
            print(f"Image not found: {item['image']}")
            continue

        # MME questions already contain "Please answer yes or no."
        answer = generate_answer(model, processor, item["image"], item["question"])

        results.append({
            "question_id": item["question_id"],
            "image": item.get("image_file", os.path.basename(item["image"])),
            "question": item["question"],
            "prediction": answer,
            "gt_answer": item["answer"],
            "category": item["category"],
            "split": item["split"],
        })

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{len(results)} answers saved to {args.output_file}")


if __name__ == "__main__":
    main()
