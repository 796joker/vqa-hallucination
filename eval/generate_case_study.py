"""
Generate qualitative case studies: Base vs SFT vs SFT+DPO on open-ended questions.
Produces structured JSON output for easy inclusion in reports/slides.

Usage:
    python eval/generate_case_study.py \
        --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
        --sft_adapter results/sft/lora_r8 \
        --dpo_adapter results/dpo/lora_r8_beta01 \
        --image_dir demo/examples \
        --output_dir results/case_studies
"""

import argparse
import json
import os
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


def load_model(base_path: str, adapter_path: str | None = None):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model


def generate(model, processor, image: Image.Image, question: str, max_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            temperature=None, top_p=None,
        )

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# Default case study questions
DEFAULT_QUESTIONS = [
    "Describe all the objects you can see in this image in detail.",
    "What is happening in this image?",
    "Is there anything unusual or out of place in this image?",
    "List every object in the scene, including small or partially visible ones.",
]


def run_case_studies(models: dict, processor, image_dir: str, output_dir: str,
                     questions: list[str] | None = None):
    os.makedirs(output_dir, exist_ok=True)
    questions = questions or DEFAULT_QUESTIONS

    # Find images
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    results = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        print(f"\n=== {img_file} ===")

        for question in questions:
            print(f"  Q: {question}")
            case = {
                "image": img_file,
                "question": question,
                "responses": {},
            }

            for model_name, model in models.items():
                answer = generate(model, processor, image, question)
                case["responses"][model_name] = answer
                # Truncate for display
                display = answer[:100] + "..." if len(answer) > 100 else answer
                print(f"    [{model_name}]: {display}")

            results.append(case)

    # Save results
    out_path = os.path.join(output_dir, "case_studies.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n{len(results)} case studies saved to {out_path}")

    # Generate markdown report
    md_path = os.path.join(output_dir, "case_studies.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Qualitative Case Studies\n\n")
        current_image = None
        for case in results:
            if case["image"] != current_image:
                current_image = case["image"]
                f.write(f"\n## {current_image}\n\n")
                f.write(f"![{current_image}](../../demo/examples/{current_image})\n\n")

            f.write(f"**Q: {case['question']}**\n\n")
            for model_name, answer in case["responses"].items():
                f.write(f"**{model_name}:**\n> {answer}\n\n")
            f.write("---\n\n")

    print(f"Markdown report saved to {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../downloads/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--sft_adapter", type=str, default="results/sft/lora_r8")
    parser.add_argument("--dpo_adapter", type=str, default="results/dpo/lora_r8_beta01")
    parser.add_argument("--image_dir", type=str, default="demo/examples")
    parser.add_argument("--output_dir", type=str, default="results/case_studies")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    print("Loading Base model...")
    base_model = load_model(args.model_path)

    print("Loading SFT model...")
    sft_model = load_model(args.model_path, args.sft_adapter)

    print("Loading SFT+DPO model...")
    dpo_model = load_model(args.model_path, args.dpo_adapter)

    models = {"Base": base_model, "SFT": sft_model, "SFT+DPO": dpo_model}
    run_case_studies(models, processor, args.image_dir, args.output_dir)
