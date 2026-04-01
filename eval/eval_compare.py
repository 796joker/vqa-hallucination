"""
Compare POPE evaluation results across Base / SFT / SFT+DPO models.
Generates comparison tables and publication-quality charts.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"


def load_metrics(eval_dir: str, splits: list[str]) -> dict:
    """Load and merge metrics from multiple POPE splits."""
    all_results = []
    for split in splits:
        path = os.path.join(eval_dir, f"pope_{split}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            all_results.extend(json.load(f))

    from eval_pope import compute_metrics
    overall = compute_metrics(all_results)

    for split in splits:
        split_results = [r for r in all_results if r["category"] == split]
        if split_results:
            split_metrics = compute_metrics(split_results)
            for k, v in split_metrics.items():
                overall[f"{split}_{k}"] = v

    return overall


def print_comparison_table(model_metrics: dict[str, dict]):
    header = f"{'Model':<25} {'Acc':<8} {'F1':<8} {'Prec':<8} {'Rec':<8} {'Yes%':<8}"
    print(header)
    print("-" * len(header))
    for name, m in model_metrics.items():
        print(
            f"{name:<25} {m['accuracy']:.4f}  {m['f1']:.4f}  "
            f"{m['precision']:.4f}  {m['recall']:.4f}  {m['yes_ratio']:.4f}"
        )


def plot_comparison(model_metrics: dict[str, dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    categories = ["random", "popular", "adversarial"]
    model_names = list(model_metrics.keys())
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    # --- Chart 1: Accuracy by category ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(categories))
    width = 0.25

    for i, (name, color) in enumerate(zip(model_names, colors)):
        accs = [model_metrics[name].get(f"{cat}_accuracy", 0) for cat in categories]
        bars = ax.bar(x + i * width, accs, width, label=name, color=color)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("POPE Category", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("POPE Accuracy: Base vs SFT vs SFT+DPO", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pope_accuracy_comparison.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "pope_accuracy_comparison.png"), dpi=300)
    plt.close()

    # --- Chart 2: Overall metrics radar ---
    fig, ax = plt.subplots(figsize=(8, 6))
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]

    x = np.arange(len(metric_names))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        values = [model_metrics[name].get(k, 0) for k in metric_keys]
        ax.bar(x + i * width, values, width, label=name, color=color)

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Overall POPE Metrics Comparison", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pope_overall_comparison.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "pope_overall_comparison.png"), dpi=300)
    plt.close()

    # --- Chart 3: Yes-ratio comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, color) in enumerate(zip(model_names, colors)):
        yes_ratios = [model_metrics[name].get(f"{cat}_yes_ratio", 0) for cat in categories]
        ax.bar(x + i * width, yes_ratios, width, label=name, color=color)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Ideal (0.5)")
    ax.set_xlabel("POPE Category", fontsize=12)
    ax.set_ylabel("Yes Ratio", fontsize=12)
    ax.set_title("Yes-Ratio by Category (lower = less hallucination)", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([c.capitalize() for c in categories])
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pope_yes_ratio.pdf"), dpi=300)
    plt.savefig(os.path.join(output_dir, "pope_yes_ratio.png"), dpi=300)
    plt.close()

    print(f"Charts saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="results/eval/base")
    parser.add_argument("--sft_dir", type=str, default="results/eval/sft")
    parser.add_argument("--dpo_dir", type=str, default="results/eval/sft_dpo")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()

    splits = ["random", "popular", "adversarial"]

    model_metrics = {
        "Base (Qwen3-VL-8B)": load_metrics(args.base_dir, splits),
        "SFT": load_metrics(args.sft_dir, splits),
        "SFT + DPO": load_metrics(args.dpo_dir, splits),
    }

    print("\n=== POPE Evaluation Comparison ===\n")
    print_comparison_table(model_metrics)
    plot_comparison(model_metrics, args.output_dir)

    # Save all metrics to JSON
    metrics_path = os.path.join(args.output_dir, "all_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(model_metrics, f, indent=2)
    print(f"\nAll metrics saved to {metrics_path}")
