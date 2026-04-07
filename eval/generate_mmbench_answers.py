"""
Generate model answers on MMBench benchmark.
MMBench: multiple-choice questions (A/B/C/D) covering 20+ ability dimensions.
Supports optional CircularEval (4 option permutations per question).

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval/generate_mmbench_answers.py \
        --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
        --adapter_path results/ablation/dpo_rlaifv_optimal_5k \
        --mmbench_dir data/mmbench \
        --output_file results/eval/dpo_rlaifv_optimal_5k/mmbench_answers.json
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


OPTION_LETTERS = ["A", "B", "C", "D"]


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


def build_prompt(question: str, hint: str, options: dict, option_order: list[str] | None = None) -> str:
    """Build multiple-choice prompt for MMBench question."""
    if option_order is None:
        option_order = [l for l in OPTION_LETTERS if l in options]

    parts = []
    if hint and hint.strip():
        parts.append(hint.strip())
    parts.append(question)
    parts.append("")  # blank line before options
    for letter in option_order:
        opt_text = options.get(letter, "")
        if opt_text:
            parts.append(f"{letter}. {opt_text}")
    parts.append("")
    parts.append("Answer with the option letter only.")

    return "\n".join(parts)


def permute_options(options: dict, shift: int) -> tuple[dict, dict]:
    """Shift option assignments for CircularEval.

    Returns:
        new_options: dict with same letters but shifted content
        mapping: dict mapping new_letter -> original_letter
    """
    available = [l for l in OPTION_LETTERS if l in options]
    n = len(available)
    if n <= 1:
        return dict(options), {l: l for l in available}

    new_options = {}
    mapping = {}  # new_letter -> original_letter
    for i, letter in enumerate(available):
        orig_idx = (i + shift) % n
        orig_letter = available[orig_idx]
        new_options[letter] = options[orig_letter]
        mapping[letter] = orig_letter
    return new_options, mapping


def parse_choice(text: str) -> str:
    """Parse model output to extract A/B/C/D choice."""
    # Strip think tags (Qwen3-VL thinking mode)
    text = re.sub(r"</?think>", "", text).strip()

    if not text:
        return "unknown"

    # Try first character
    first = text[0].upper()
    if first in OPTION_LETTERS:
        return first

    # Try to find pattern like "A." or "(A)" or "Answer: A"
    match = re.search(r"[(\s]?([A-D])[).\s:,]", text.upper())
    if match:
        return match.group(1)

    # Try to find standalone letter
    match = re.search(r"\b([A-D])\b", text.upper())
    if match:
        return match.group(1)

    return "unknown"


def generate_answer(model, processor, image_path: str, prompt: str) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
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

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    answer = processor.decode(generated, skip_special_tokens=True)
    answer = re.sub(r"</?think>", "", answer).strip()
    return answer


def load_questions(mmbench_dir: str) -> list[dict]:
    json_path = os.path.join(mmbench_dir, "questions.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"questions.json not found in {mmbench_dir}. "
            f"Run data/prepare_mmbench.py first."
        )
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../downloads/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--mmbench_dir", type=str, default="data/mmbench")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--circular_eval", action="store_true",
                        help="Run 4 permutations per question for CircularEval")
    args = parser.parse_args()

    # Skip if output already exists
    if os.path.exists(args.output_file):
        print(f"Output already exists: {args.output_file}")
        return

    questions = load_questions(args.mmbench_dir)
    print(f"Loaded {len(questions)} questions from {args.mmbench_dir}")

    model, processor = load_model(args.model_path, args.adapter_path)

    results = []
    n_permutations = 4 if args.circular_eval else 1
    total = len(questions) * n_permutations
    desc = f"MMBench {'CircularEval' if args.circular_eval else 'eval'}"

    with tqdm(total=total, desc=desc) as pbar:
        for item in questions:
            if not os.path.exists(item["image"]):
                print(f"Image not found: {item['image']}")
                pbar.update(n_permutations)
                continue

            options = item.get("options", {})
            if not options:
                pbar.update(n_permutations)
                continue

            for perm_idx in range(n_permutations):
                if perm_idx == 0:
                    cur_options = options
                    mapping = {l: l for l in OPTION_LETTERS}
                else:
                    cur_options, mapping = permute_options(options, perm_idx)

                prompt = build_prompt(
                    item["question"],
                    item.get("hint", ""),
                    cur_options,
                )
                raw_answer = generate_answer(model, processor, item["image"], prompt)
                parsed = parse_choice(raw_answer)

                # Map back to original option letter
                if parsed in mapping:
                    original_choice = mapping[parsed]
                else:
                    original_choice = "unknown"

                entry = {
                    "question_id": item["question_id"],
                    "image": item.get("image_file", os.path.basename(item["image"])),
                    "question": item["question"],
                    "raw_prediction": raw_answer,
                    "prediction": original_choice,
                    "gt_answer": item.get("answer", ""),
                    "category": item.get("category", ""),
                    "l2_category": item.get("l2_category", ""),
                }

                if args.circular_eval:
                    entry["permutation"] = perm_idx
                    entry["permuted_prediction"] = parsed

                results.append(entry)
                pbar.update(1)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{len(results)} predictions saved to {args.output_file}")
    if args.circular_eval:
        print(f"  ({len(questions)} questions x {n_permutations} permutations)")


if __name__ == "__main__":
    main()
