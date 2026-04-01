"""
Generate model answers on POPE benchmark.
Supports: base model, SFT model (with adapter), SFT+DPO model (with adapter).
"""

import argparse
import json
import os
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


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
                {"type": "text", "text": f"{question} Please answer with yes or no."},
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
            max_new_tokens=32,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = output_ids[0][inputs.input_ids.shape[1] :]
    answer = processor.decode(generated, skip_special_tokens=True)
    return answer.strip()


def run_pope_evaluation(model, processor, pope_file: str, output_file: str):
    with open(pope_file, "r") as f:
        questions = json.load(f)

    results = []
    for item in tqdm(questions, desc=f"Evaluating {os.path.basename(pope_file)}"):
        if not os.path.exists(item["image"]):
            continue

        answer = generate_answer(model, processor, item["image"], item["question"])
        results.append({
            "question_id": item["question_id"],
            "image": item["image_file"],
            "question": item["question"],
            "answer": answer,
            "gt_answer": item["answer"],
            "category": item["category"],
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(results)} answers -> {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../downloads/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--pope_dir", type=str, default="data/pope_data")
    parser.add_argument("--output_dir", type=str, required=True, help="e.g. results/eval/base")
    parser.add_argument("--split", type=str, default=None,
                        choices=["random", "popular", "adversarial"],
                        help="Run only this split (default: all three)")
    args = parser.parse_args()

    model, processor = load_model(args.model_path, args.adapter_path)

    if args.split:
        pope_splits = [f"pope_{args.split}.json"]
    else:
        pope_splits = ["pope_random.json", "pope_popular.json", "pope_adversarial.json"]

    for split_file in pope_splits:
        pope_path = os.path.join(args.pope_dir, split_file)
        if not os.path.exists(pope_path):
            print(f"Skipping {pope_path} (not found)")
            continue
        output_path = os.path.join(args.output_dir, split_file)
        if os.path.exists(output_path):
            print(f"Skipping {split_file} (already exists at {output_path})")
            continue
        run_pope_evaluation(model, processor, pope_path, output_path)
