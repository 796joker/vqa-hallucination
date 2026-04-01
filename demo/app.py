"""
Gradio demo: Side-by-side comparison of Base vs SFT+DPO model.
Upload an image and ask a question to see both models' answers.
"""

import argparse
import torch
import gradio as gr
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info


def load_model(base_model_path: str, adapter_path: str | None = None, use_4bit: bool = False):
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if use_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(base_model_path, **kwargs)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model


def generate(model, processor, image: Image.Image, question: str, max_tokens: int = 512) -> str:
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
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated = output_ids[0][inputs.input_ids.shape[1] :]
    return processor.decode(generated, skip_special_tokens=True).strip()


def create_demo(base_model, ft_model, processor):
    def compare(image, question):
        if image is None:
            return "Please upload an image.", ""
        if not question.strip():
            return "Please enter a question.", ""

        image = Image.fromarray(image).convert("RGB")
        base_answer = generate(base_model, processor, image, question)
        ft_answer = generate(ft_model, processor, image, question)
        return base_answer, ft_answer

    def single_query(image, question):
        if image is None:
            return "Please upload an image."
        if not question.strip():
            return "Please enter a question."

        image = Image.fromarray(image).convert("RGB")
        return generate(ft_model, processor, image, question)

    with gr.Blocks(
        title="VQA Hallucination Reduction Demo",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# Reducing Visual Hallucinations with SFT + DPO\n"
            "Compare **Qwen3-VL-8B Base** vs **SFT+DPO Fine-tuned** model on visual question answering."
        )

        with gr.Tab("Side-by-Side Comparison"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Image", type="numpy")
                    q_input = gr.Textbox(
                        label="Question",
                        placeholder="e.g., Describe all objects in this image.",
                        lines=2,
                    )
                    compare_btn = gr.Button("Compare Both Models", variant="primary")

                with gr.Column(scale=1):
                    base_output = gr.Textbox(label="Base Model (Qwen3-VL-8B)", lines=10)
                with gr.Column(scale=1):
                    ft_output = gr.Textbox(label="SFT + DPO Model", lines=10)

            compare_btn.click(
                fn=compare,
                inputs=[img_input, q_input],
                outputs=[base_output, ft_output],
            )

            gr.Examples(
                examples=[
                    ["demo/examples/example1.jpg", "Describe all the objects you can see in this image."],
                    ["demo/examples/example2.jpg", "Is there a cat in this image?"],
                    ["demo/examples/example3.jpg", "What is the person doing in this image?"],
                ],
                inputs=[img_input, q_input],
            )

        with gr.Tab("Single Model Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    single_img = gr.Image(label="Upload Image", type="numpy")
                    single_q = gr.Textbox(label="Question", lines=2)
                    single_btn = gr.Button("Ask", variant="primary")
                with gr.Column(scale=1):
                    single_output = gr.Textbox(label="SFT + DPO Model Answer", lines=12)

            single_btn.click(
                fn=single_query,
                inputs=[single_img, single_q],
                outputs=[single_output],
            )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../downloads/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="results/dpo/lora_r8_beta01")
    parser.add_argument("--use_4bit", action="store_true", help="Load models in 4-bit to save memory")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    print("Loading base model...")
    base_model = load_model(args.model_path, use_4bit=args.use_4bit)

    print("Loading fine-tuned model...")
    ft_model = load_model(args.model_path, args.adapter_path, use_4bit=args.use_4bit)

    print("Starting Gradio demo...")
    demo = create_demo(base_model, ft_model, processor)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
