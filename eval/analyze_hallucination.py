"""
Fine-grained hallucination type analysis.

Analyzes POPE results to identify:
1. Most commonly hallucinated objects
2. False positive (hallucination) vs false negative (miss) breakdown
3. Comparison across models
"""

import argparse
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"


def parse_yesno(text: str) -> str:
    text = text.strip().lower()
    return "yes" if text.startswith("yes") else "no"


def analyze_hallucinations(result_file: str) -> dict:
    with open(result_file) as f:
        results = json.load(f)

    # False positives: model says "yes" but ground truth is "no" (hallucination)
    hallucinations = [
        r for r in results
        if r["gt_answer"] == "no" and parse_yesno(r["answer"]) == "yes"
    ]

    # False negatives: model says "no" but ground truth is "yes" (miss)
    misses = [
        r for r in results
        if r["gt_answer"] == "yes" and parse_yesno(r["answer"]) == "no"
    ]

    # Extract hallucinated objects from POPE question format
    hallucinated_objects = []
    for r in hallucinations:
        q = r["question"]
        # POPE format: "Is there a/an [object] in the image?"
        for prefix in ["Is there a ", "Is there an "]:
            if prefix in q:
                obj = q.split(prefix)[1].split(" in the image")[0]
                hallucinated_objects.append(obj)
                break

    return {
        "total": len(results),
        "hallucinations": len(hallucinations),
        "hallucination_rate": len(hallucinations) / len(results) if results else 0,
        "misses": len(misses),
        "miss_rate": len(misses) / len(results) if results else 0,
        "hallucinated_objects": Counter(hallucinated_objects).most_common(30),
        "examples": hallucinations[:10],
    }


def compare_hallucinations(model_dirs: dict[str, str], output_dir: str):
    """Compare hallucination patterns across models."""
    os.makedirs(output_dir, exist_ok=True)
    splits = ["pope_random.json", "pope_popular.json", "pope_adversarial.json"]

    all_analyses = {}
    for model_name, eval_dir in model_dirs.items():
        model_analysis = {}
        for split_file in splits:
            path = os.path.join(eval_dir, split_file)
            if os.path.exists(path):
                split_name = split_file.replace("pope_", "").replace(".json", "")
                model_analysis[split_name] = analyze_hallucinations(path)
        all_analyses[model_name] = model_analysis

    # Print summary
    print("\n=== Hallucination Analysis ===\n")
    for model_name, splits_data in all_analyses.items():
        print(f"\n--- {model_name} ---")
        for split_name, analysis in splits_data.items():
            print(f"  {split_name}:")
            print(f"    Hallucination rate: {analysis['hallucination_rate']:.4f} ({analysis['hallucinations']}/{analysis['total']})")
            print(f"    Miss rate:          {analysis['miss_rate']:.4f} ({analysis['misses']}/{analysis['total']})")
            if analysis["hallucinated_objects"]:
                top5 = ", ".join(f"{obj}({cnt})" for obj, cnt in analysis["hallucinated_objects"][:5])
                print(f"    Top hallucinated:   {top5}")

    # Plot hallucination rate comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["random", "popular", "adversarial"]
    model_names = list(all_analyses.keys())
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    x = np.arange(len(categories))
    width = 0.25

    import numpy as np

    for i, (name, color) in enumerate(zip(model_names, colors)):
        rates = []
        for cat in categories:
            if cat in all_analyses[name]:
                rates.append(all_analyses[name][cat]["hallucination_rate"])
            else:
                rates.append(0)
        bars = ax.bar(x + i * width, rates, width, label=name, color=color)
        for bar, val in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("POPE Category", fontsize=12)
    ax.set_ylabel("Hallucination Rate", fontsize=12)
    ax.set_title("Hallucination Rate Comparison (lower is better)", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hallucination_rate_comparison.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "hallucination_rate_comparison.png"), dpi=300)
    plt.close()

    # Plot top hallucinated objects for SFT+DPO model (adversarial split)
    dpo_name = model_names[-1]
    if "adversarial" in all_analyses[dpo_name]:
        objects = all_analyses[dpo_name]["adversarial"]["hallucinated_objects"][:15]
        if objects:
            fig, ax = plt.subplots(figsize=(10, 6))
            names = [o[0] for o in objects]
            counts = [o[1] for o in objects]
            ax.barh(range(len(names)), counts, color="#C44E52")
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.set_xlabel("Count", fontsize=12)
            ax.set_title(f"Top Hallucinated Objects ({dpo_name}, Adversarial)", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "top_hallucinated_objects.pdf"), dpi=300)
            plt.savefig(os.path.join(output_dir, "top_hallucinated_objects.png"), dpi=300)
            plt.close()

    print(f"\nCharts saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results/eval/base")
    parser.add_argument("--sft_dir", type=str, default="results/eval/sft")
    parser.add_argument("--dpo_dir", type=str, default="results/eval/sft_dpo")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()

    model_dirs = {
        "Base": args.base_dir,
        "SFT": args.sft_dir,
        "SFT + DPO": args.dpo_dir,
    }
    compare_hallucinations(model_dirs, args.output_dir)
