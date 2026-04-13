"""
Gradio Demo: VQA Hallucination Mitigation via SFT + DPO
视觉问答幻觉缓解演示 — 基座模型 vs True Optimal 对比

Course: 大模型后训练技术
Model: Qwen3-VL-8B-Instruct + LoRA (SFT 5K + DPO β=1.0 1ep)
"""

import re
import os
import sys
import random
import argparse
import torch
import gradio as gr
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from typing import Optional


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(base_model_path: str, adapter_path: Optional[str] = None, use_4bit: bool = False):
    """Load base model, optionally merge a LoRA adapter."""
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
        print(f"  Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

THINK_TAG_RE = re.compile(r"</?think>")
YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


def parse_yesno(text: str) -> str:
    """Extract yes/no from model output using word boundary matching."""
    text_lower = text.lower().strip()
    if text_lower.startswith("yes"):
        return "yes"
    if text_lower.startswith("no"):
        return "no"
    if YES_RE.search(text):
        return "yes"
    if NO_RE.search(text):
        return "no"
    return "unknown"


def clean_output(text: str) -> str:
    """Strip DPO think-tag artifacts and normalize whitespace."""
    text = THINK_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate(model, processor, image: Image.Image, question: str,
             max_tokens: int = 512) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    raw = processor.decode(generated, skip_special_tokens=True)
    return clean_output(raw)


# ---------------------------------------------------------------------------
# Experiment metrics (verified from server DATA_VERIFICATION.md)
# ---------------------------------------------------------------------------

METRICS_TABLE = """
| 模型配置 | POPE F1 | CHAIR_i | MME 总分 | MME CPR | 训练时间 |
|----------|---------|---------|----------|---------|----------|
| **Base (无训练)** | 0.906 | 33.31% | 2008.0 | 100.0% | — |
| SFT 50K | 0.895 | 16.64% | — | — | 5h |
| SFT 5K | **0.922** | 16.73% | 1899.0 | 94.6% | 0.5h |
| DPO-only | 0.900 | 31.83% | 1964.5 | 97.8% | 0.5h |
| **True Optimal** | 0.889 | **20.12%** | **1990.5** | **99.1%** | 1h |

> 注：MME 仅评估 4 个关键模型；所有 POPE F1 为 Random split。
> 数据来源：服务器 results/eval/ 原始预测文件重新计算，详见 DATA_VERIFICATION.md。
"""

ABLATION_TABLE = """
### LoRA 秩消融 (r = 4, 8, 16, 32)
- F1 波动仅 <2pp，r=8 即足够

### SFT 数据规模 — "少即是多"
| 数据量 | POPE F1 (R) | POPE F1 (P) | POPE F1 (A) | Yes-Ratio (R) | 训练时间 |
|--------|-------------|-------------|-------------|---------------|---------|
| 5K | **0.922** | **0.893** | **0.863** | 0.457 | 0.5h |
| 10K | 0.903 | 0.879 | 0.850 | 0.446 | 1h |
| 25K | 0.893 | 0.858 | 0.828 | 0.456 | 2.5h |
| 50K | 0.895 | 0.856 | 0.823 | 0.469 | 5h |

### DPO Beta 敏感性
| β | POPE F1 | Yes-Ratio | 状态 |
|---|---------|-----------|------|
| 0.01 | 0.000 | 0.000 | 崩溃 |
| 0.05 | 0.076 | 0.020 | 崩溃 |
| 0.1 | 0.780 | 0.322 | 正常 |
| 0.5 | 0.841 | 0.370 | 正常 |
| **1.0** | **0.846** | **0.374** | 最优 |

### 训练轮数
- 1 轮 > 3 轮：+8.9pp F1 (0.869 vs 0.780)

### 损失函数
- Hinge (0.791) ≈ Sigmoid (0.780) > IPO (崩溃)
- Sigmoid 在高 beta 下表现更优，综合最稳健
"""

KEY_FINDINGS = """
## 四大核心发现

### 1. "少即是多" — SFT 数据规模
5K 数据 POPE F1=0.922，优于 50K 的 0.895 (+2.7pp)，训练快 **10 倍**。
在 Popular/Adversarial split 上差距更大 (+3.7pp / +4.0pp)。

### 2. 知识灾难性遗忘
SFT 导致知识任务下降 **-7.03pp**（名人 -7.35pp，艺术品 -7.00pp）。
DPO 恢复：名人识别超基线 +2.65pp。

### 3. True Optimal 配置
SFT 5K + DPO β=1.0 1 轮 → CHAIR_i 降低 **39.6%**，能力保持 **99.1%**。
训练 1 小时，成本 ~$2.50。

### 4. DPO-only 悖论
POPE F1=0.900（最佳判别）但 CHAIR_i=31.83%（接近 Base）。
判别质量 ≠ 生成质量，证明 SFT 不可跳过。
"""


# ---------------------------------------------------------------------------
# Preset questions for demo
# ---------------------------------------------------------------------------

PRESET_QUESTIONS = {
    "详细描述": "Please describe this image in detail.",
    "物体存在性 (POPE)": "Is there a person in this image? Answer yes or no.",
    "动物检测": "Is there a cat in this image? Answer yes or no.",
    "家具检测": "Is there a chair in this image? Answer yes or no.",
    "车辆检测": "Is there a car in this image? Answer yes or no.",
    "物体计数": "How many people are in this image?",
    "属性识别": "What color is the main object in this image?",
    "空间关系": "Describe the spatial relationship between the objects in this image.",
    "自定义问题": "",
}

EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")


def get_example_images(limit=6):
    """Return list of valid example image paths (skip broken symlinks)."""
    if not os.path.isdir(EXAMPLE_DIR):
        return []
    result = []
    for f in sorted(os.listdir(EXAMPLE_DIR)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(EXAMPLE_DIR, f)
            if os.path.isfile(path) and os.path.getsize(path) > 1000:
                result.append(path)
    return result[:limit]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_demo(base_model, ft_model, processor):

    # --- Tab 1: Side-by-Side Comparison ---
    def compare(image, preset, custom_question):
        if image is None:
            return "请上传图片", "", ""
        question = custom_question.strip() if preset == "自定义问题" else PRESET_QUESTIONS[preset]
        if not question:
            return "请输入问题", "", ""

        pil_image = Image.fromarray(image).convert("RGB")
        base_ans = generate(base_model, processor, pil_image, question)
        ft_ans = generate(ft_model, processor, pil_image, question)

        analysis = _analyze_diff(question, base_ans, ft_ans)
        return base_ans, ft_ans, analysis

    def _analyze_diff(question: str, base: str, ft: str) -> str:
        lines = []
        is_yesno = any(kw in question.lower() for kw in ["yes or no", "是否", "有没有"])
        if is_yesno:
            b = parse_yesno(base)
            f = parse_yesno(ft)
            if b != f:
                lines.append(f"**判别差异**: Base={b}, True Optimal={f}")
            else:
                lines.append(f"**判别一致**: 均回答 {b}")
                if b == "yes" and f == "yes":
                    lines.append("注意: 两个模型都回答 yes — 若实际不存在该物体，则为共同幻觉")
        b_len, f_len = len(base.split()), len(ft.split())
        if b_len > 0:
            ratio = f_len / b_len
            if ratio < 0.7:
                lines.append(f"True Optimal 回答更简洁 ({f_len} vs {b_len} 词)")
            elif ratio > 1.3:
                lines.append(f"True Optimal 回答更详细 ({f_len} vs {b_len} 词)")
        return "\n".join(lines) if lines else "两个模型回答相近"

    # --- Tab 2: POPE Hallucination Batch Test ---
    POPE_OBJECTS = [
        "person", "car", "chair", "dog", "cat", "bus", "train",
        "laptop", "pizza", "clock", "vase", "bird", "horse",
        "airplane", "umbrella", "bottle", "bowl", "banana",
    ]

    def pope_test(image):
        if image is None:
            return "请上传图片"
        pil_image = Image.fromarray(image).convert("RGB")
        random.seed(42)
        objects = random.sample(POPE_OBJECTS, min(8, len(POPE_OBJECTS)))

        rows = ["| 物体 | Base 模型 | True Optimal | 差异 |",
                "|------|-----------|-------------|------|"]
        base_yes = 0
        ft_yes = 0
        for obj in objects:
            q = f"Is there a {obj} in this image? Answer yes or no."
            b = generate(base_model, processor, pil_image, q, max_tokens=16)
            f = generate(ft_model, processor, pil_image, q, max_tokens=16)
            b_yn = parse_yesno(b)
            f_yn = parse_yesno(f)
            if b_yn == "yes":
                base_yes += 1
            if f_yn == "yes":
                ft_yes += 1
            diff = "一致" if b_yn == f_yn else "不同"
            rows.append(f"| {obj} | {b_yn} | {f_yn} | {diff} |")

        total = len(objects)
        rows.append("")
        rows.append(f"**Base Yes-Ratio**: {base_yes}/{total} = {base_yes/total:.1%}")
        rows.append(f"**True Optimal Yes-Ratio**: {ft_yes}/{total} = {ft_yes/total:.1%}")
        if base_yes > ft_yes:
            rows.append(f"True Optimal 更保守，减少了 {base_yes - ft_yes} 个潜在幻觉")
        return "\n".join(rows)

    # --- Tab 3: Caption Comparison (CHAIR-style) ---
    def caption_compare(image):
        if image is None:
            return "请上传图片", "", ""
        pil_image = Image.fromarray(image).convert("RGB")
        prompt = "Please describe this image in detail."
        base_cap = generate(base_model, processor, pil_image, prompt, max_tokens=256)
        ft_cap = generate(ft_model, processor, pil_image, prompt, max_tokens=256)
        b_words = len(base_cap.split())
        f_words = len(ft_cap.split())
        analysis = (
            f"**Base**: {b_words} 词 | **True Optimal**: {f_words} 词\n\n"
            f"实验数据参考: Base CHAIR_i=33.31% (每 3 个物体约 1 个幻觉), "
            f"True Optimal CHAIR_i=20.12% (幻觉减少 39.6%)"
        )
        return base_cap, ft_cap, analysis

    # --- Build UI ---
    with gr.Blocks(
        title="VQA 幻觉缓解演示 | SFT + DPO",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 视觉问答幻觉缓解演示\n"
            "### 基于 SFT + DPO 的 Qwen3-VL-8B 后训练\n"
            "对比 **基座模型** 与 **True Optimal** (SFT 5K + DPO β=1.0 1ep) 的回答差异\n\n"
            "---"
        )

        example_images = get_example_images()
        if not example_images:
            gr.Markdown(
                "> **提示**: 未检测到示例图片。运行 `python demo/download_examples.py` 下载，"
                "或手动上传图片使用。"
            )

        # Tab 1: Side-by-Side
        with gr.Tab("对比演示"):
            gr.Markdown("上传图片，选择问题类型，对比两个模型的回答。")
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="上传图片", type="numpy")
                    preset_dropdown = gr.Dropdown(
                        choices=list(PRESET_QUESTIONS.keys()),
                        value="详细描述",
                        label="预设问题",
                    )
                    custom_q = gr.Textbox(
                        label="自定义问题（选择「自定义问题」时填写）",
                        placeholder="e.g., Is there a dog in this image?",
                        lines=2,
                    )
                    compare_btn = gr.Button("对比两个模型", variant="primary", size="lg")

                with gr.Column(scale=1):
                    base_output = gr.Textbox(label="Base 模型 (Qwen3-VL-8B)", lines=10)
                with gr.Column(scale=1):
                    ft_output = gr.Textbox(label="True Optimal (SFT+DPO)", lines=10)

            analysis_output = gr.Markdown(label="差异分析")
            compare_btn.click(
                fn=compare,
                inputs=[img_input, preset_dropdown, custom_q],
                outputs=[base_output, ft_output, analysis_output],
            )

            if example_images:
                gr.Examples(
                    examples=[[img] for img in example_images],
                    inputs=[img_input],
                    label="示例图片（点击直接使用）",
                )

            with gr.Accordion("边界案例提示（展示模型局限性）", open=False):
                gr.Markdown("""
**以下场景容易触发幻觉，适合展示模型局限性：**

1. **共现混淆**: 图中有卡车 → 问 "Is there a car?"（高共现物体对容易误判）
2. **小/遮挡物体**: 询问远处或被部分遮挡的物体（背景中的椅子、远处的时钟）
3. **相似物体**: 滑雪板 vs 雪橇、叉子 vs 勺子
4. **复杂场景**: 餐桌（多物体堆叠）、街景（多类别物体密集）
5. **计数任务**: "How many people are in this image?" — 两个模型都容易出错

**展示技巧：** 两个模型都回答 "yes" 时，可能是共同幻觉（POPE 评估中约 11% 的对抗样本双模型均错误）。True Optimal 更保守（yes-ratio 0.413 vs Base 0.431），偶尔会过度保守导致漏检。
""")

        # Tab 2: POPE-style Hallucination Test
        with gr.Tab("幻觉检测 (POPE)"):
            gr.Markdown(
                "上传图片后，自动用 8 个常见物体进行 POPE 风格的存在性测试，"
                "对比 Base 模型与 True Optimal 的 yes-bias 差异。"
            )
            with gr.Row():
                pope_img = gr.Image(label="上传图片", type="numpy")
                pope_result = gr.Markdown(label="POPE 检测结果")
            pope_btn = gr.Button("运行幻觉检测", variant="primary")
            pope_btn.click(fn=pope_test, inputs=[pope_img], outputs=[pope_result])

            if example_images:
                gr.Examples(
                    examples=[[img] for img in example_images[:4]],
                    inputs=[pope_img],
                    label="示例图片",
                )

        # Tab 3: Caption Comparison (CHAIR-style)
        with gr.Tab("描述对比 (CHAIR)"):
            gr.Markdown(
                "上传图片，让两个模型分别生成详细描述，对比幻觉差异。\n"
                "Base 模型通常生成更长但含更多幻觉的描述，True Optimal 更简洁准确。"
            )
            with gr.Row():
                caption_img = gr.Image(label="上传图片", type="numpy")
            caption_btn = gr.Button("生成描述对比", variant="primary")
            with gr.Row():
                caption_base = gr.Textbox(label="Base 模型描述", lines=8)
                caption_ft = gr.Textbox(label="True Optimal 描述", lines=8)
            caption_analysis = gr.Markdown(label="对比分析")
            caption_btn.click(
                fn=caption_compare,
                inputs=[caption_img],
                outputs=[caption_base, caption_ft, caption_analysis],
            )

            if example_images:
                gr.Examples(
                    examples=[[img] for img in example_images[:4]],
                    inputs=[caption_img],
                    label="示例图片",
                )

        # Tab 4: Metrics Dashboard
        with gr.Tab("实验指标"):
            gr.Markdown("## 核心模型对比\n" + METRICS_TABLE)
            gr.Markdown("---")
            gr.Markdown(KEY_FINDINGS)

        # Tab 5: Ablation Results
        with gr.Tab("消融实验"):
            gr.Markdown("## 五维度消融结果\n" + ABLATION_TABLE)

        # Tab 6: About
        with gr.Tab("关于"):
            gr.Markdown("""
## 项目信息

| 项目 | 详情 |
|------|------|
| **课程** | 大模型后训练技术 |
| **基座模型** | Qwen3-VL-8B-Instruct |
| **SFT 数据** | LLaVA-Instruct-150K (5K 子集) |
| **DPO 数据** | RLHF-V (5,733 偏好对, 人工标注) |
| **训练框架** | LLaMA-Factory 0.9.5 + Accelerate DDP |
| **硬件** | NVIDIA A100-SXM4-80GB (共享服务器) |
| **评估基准** | POPE (9K) + CHAIR (500) + MME (2374) |
| **消融规模** | 5 维度，20 个模型配置 |

### True Optimal 配置
```
SFT:  5K data, LoRA r=8, 2 epochs  (30 min)
DPO:  β=1.0, 1 epoch, sigmoid loss (30 min)
Total: 1 hour, ~$2.50
```

### 服务器环境
- Ubuntu 22.04 / Python 3.12 / PyTorch 2.10 / CUDA 12.4
- Transformers 4.57.6 / PEFT 0.18.1
- 代码: https://github.com/796joker/vqa-hallucination
""")

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQA Hallucination Demo")
    parser.add_argument("--model_path", type=str,
                        default=os.environ.get("MODEL_PATH", "models/Qwen3-VL-8B-Instruct"),
                        help="Path to base model (or set MODEL_PATH env var)")
    parser.add_argument("--adapter_path", type=str,
                        default=os.environ.get("ADAPTER_PATH", "results/ablation/dpo_true_optimal"),
                        help="Path to True Optimal adapter (or set ADAPTER_PATH env var)")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Load models in 4-bit quantization (saves ~20GB VRAM)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("GRADIO_PORT", "6006")))
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    # Validate paths before loading
    if not os.path.isdir(args.model_path):
        print(f"ERROR: Base model not found at: {args.model_path}")
        print("Please download first:")
        print("  hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct")
        print("Or specify via: --model_path /path/to/model  or  MODEL_PATH=/path/to/model")
        sys.exit(1)
    if not os.path.isdir(args.adapter_path):
        print(f"ERROR: Adapter not found at: {args.adapter_path}")
        print("Please ensure the repo is cloned with adapter weights.")
        print("Or specify via: --adapter_path /path/to/adapter  or  ADAPTER_PATH=/path/to/adapter")
        sys.exit(1)

    print("=" * 60)
    print("VQA Hallucination Mitigation Demo")
    print("=" * 60)
    print(f"  Base model:  {os.path.abspath(args.model_path)}")
    print(f"  Adapter:     {os.path.abspath(args.adapter_path)}")
    print(f"  4-bit:       {args.use_4bit}")
    print(f"  Port:        {args.port}")
    valid_examples = get_example_images()
    if not valid_examples:
        print("  Examples:    NONE (run: python demo/download_examples.py)")
    else:
        print(f"  Examples:    {len(valid_examples)} images")
    if torch.cuda.is_available():
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  VRAM:        {vram_gb:.1f} GB")
        if vram_gb < 20 and not args.use_4bit:
            print(f"  WARNING: VRAM < 20GB, consider using --use_4bit")
    else:
        print("  WARNING: No CUDA GPU detected, model loading will be very slow")
    print("=" * 60)

    try:
        print(f"\n[1/3] Loading processor from {args.model_path} ...")
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

        print(f"[2/3] Loading base model ...")
        base_model = load_model(args.model_path, use_4bit=args.use_4bit)

        print(f"[3/3] Loading True Optimal model (adapter: {args.adapter_path}) ...")
        ft_model = load_model(args.model_path, args.adapter_path, use_4bit=args.use_4bit)
    except torch.cuda.OutOfMemoryError:
        print("\nERROR: GPU out of memory!")
        print("Try: python demo/app.py --use_4bit")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR loading models: {e}")
        sys.exit(1)

    print(f"\nModels loaded. Starting Gradio on port {args.port} ...")
    demo = create_demo(base_model, ft_model, processor)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
