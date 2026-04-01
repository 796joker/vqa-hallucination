"""
Generate image captions for CHAIR evaluation.
Uses 500 COCO val2014 images, prompts model with "Please describe this image in detail."

Usage:
    python eval/generate_chair_captions.py \
        --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
        --adapter_path results/sft/lora_r8 \
        --image_dir ../downloads/coco/val2014 \
        --output_file results/eval/sft/chair_captions.json
"""

import argparse
import json
import os
import random

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


CAPTION_PROMPT = "Please describe this image in detail."


def load_model(base_path: str, adapter_path: str | None = None):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model


def generate_caption(model, processor, image: Image.Image,
                     max_tokens: int = 512) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": CAPTION_PROMPT},
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, temperature=None, top_p=None,
        )

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="../downloads/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--image_dir", type=str,
                        default="../downloads/coco/val2014")
    parser.add_argument("--image_list", type=str, default=None,
                        help="JSON file with list of image filenames to use")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=500)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Get image list
    if args.image_list:
        with open(args.image_list) as f:
            image_files = json.load(f)
    else:
        image_files = sorted([
            f for f in os.listdir(args.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        random.shuffle(image_files)
        image_files = image_files[:args.num_images]

    print(f"Will caption {len(image_files)} images from {args.image_dir}")

    # Load model
    print(f"Loading model from {args.model_path}")
    if args.adapter_path:
        print(f"  with adapter: {args.adapter_path}")
    model = load_model(args.model_path, args.adapter_path)
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)

    # Generate captions
    results = []
    for img_file in tqdm(image_files, desc="Generating captions"):
        img_path = os.path.join(args.image_dir, img_file)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [SKIP] {img_file}: {e}")
            continue

        # Extract image_id from filename (e.g., COCO_val2014_000000XXXXXX.jpg)
        image_id = img_file.split(".")[0]
        # Try to extract numeric ID
        parts = image_id.split("_")
        if parts:
            try:
                image_id = str(int(parts[-1]))
            except ValueError:
                pass

        caption = generate_caption(model, processor, image, args.max_tokens)
        results.append({
            "image_id": image_id,
            "image_file": img_file,
            "caption": caption,
        })

    # Save
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n{len(results)} captions saved to {args.output_file}")


if __name__ == "__main__":
    main()
