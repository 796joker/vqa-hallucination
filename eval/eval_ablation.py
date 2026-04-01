"""
Comprehensive ablation experiment analysis and visualization.
Generates publication-quality charts for each ablation group.

Usage:
    python eval/eval_ablation.py --eval_root results/eval --output_dir results/figures/ablation
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 11

# Add parent dir so we can import eval_pope
sys.path.insert(0, os.path.dirname(__file__))
from eval_pope import compute_metrics


# ============================================================
# Data Loading
# ============================================================

def load_pope_metrics(eval_dir: str) -> dict | None:
    """Load POPE evaluation results from a directory."""
    splits = ["pope_random.json", "pope_popular.json", "pope_adversarial.json"]
    all_results = []

    for split_file in splits:
        path = os.path.join(eval_dir, split_file)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            all_results.extend(json.load(f))

    if not all_results:
        return None

    metrics = compute_metrics(all_results)

    # Per-split metrics
    for split_name in ["random", "popular", "adversarial"]:
        split_results = [r for r in all_results if r.get("category") == split_name]
        if split_results:
            split_m = compute_metrics(split_results)
            for k, v in split_m.items():
                metrics[f"{split_name}_{k}"] = v

    return metrics


def collect_ablation_metrics(eval_root: str, prefix: str) -> dict[str, dict]:
    """Collect metrics for all eval dirs matching a prefix pattern."""
    results = {}
    eval_root = Path(eval_root)
    for d in sorted(eval_root.iterdir()):
        if d.is_dir() and d.name.startswith(prefix):
            metrics = load_pope_metrics(str(d))
            if metrics:
                results[d.name] = metrics
    return results


# ============================================================
# Plotting Utilities
# ============================================================

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
          "#64B5CD", "#D65F5F", "#4878CF", "#6ACC65", "#D5BB67"]


def save_fig(fig, path_base: str):
    """Save figure as both PDF and PNG."""
    fig.savefig(f"{path_base}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_line(x_values, y_dict: dict[str, list], xlabel: str, ylabel: str,
              title: str, output_path: str, x_labels=None, ideal_line=None):
    """Line plot for continuous variable ablation."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (label, values) in enumerate(y_dict.items()):
        color = COLORS[i % len(COLORS)]
        ax.plot(range(len(x_values)), values, 'o-', color=color, label=label,
                linewidth=2, markersize=8)
        for j, v in enumerate(values):
            ax.annotate(f"{v:.3f}", (j, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    if ideal_line is not None:
        ax.axhline(y=ideal_line, color="gray", linestyle="--", alpha=0.5, label="Ideal")

    if x_labels:
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels(x_labels)
    else:
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([str(v) for v in x_values])

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, output_path)


def plot_grouped_bar(data: dict[str, dict[str, float]], xlabel: str, ylabel: str,
                     title: str, output_path: str):
    """Grouped bar chart for categorical variable ablation."""
    categories = list(data.keys())
    metrics = list(next(iter(data.values())).keys())

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.5), 5))
    x = np.arange(len(categories))
    width = 0.8 / len(metrics)

    for i, metric in enumerate(metrics):
        values = [data[cat].get(metric, 0) for cat in categories]
        bars = ax.bar(x + i * width - 0.4 + width / 2, values, width,
                      label=metric, color=COLORS[i % len(COLORS)])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15 if len(categories) > 4 else 0)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_fig(fig, output_path)


# ============================================================
# Ablation Group Analysis Functions
# ============================================================

def analyze_pipeline(eval_root: str, out_dir: str):
    """G1: Base vs SFT vs DPO-only vs SFT+DPO."""
    labels = {
        "base": "Base",
        "sft": "SFT",
        "dpo_only": "DPO-only",
        "dpo": "SFT+DPO",
    }
    data = {}
    for key, label in labels.items():
        d = os.path.join(eval_root, key)
        m = load_pope_metrics(d)
        if m:
            data[label] = m

    if len(data) < 2:
        print(f"  [SKIP] Pipeline: only {len(data)} models found")
        return

    # Grouped bar: overall metrics
    bar_data = {name: {"Accuracy": m["accuracy"], "Precision": m["precision"],
                       "Recall": m["recall"], "F1": m["f1"]}
                for name, m in data.items()}
    plot_grouped_bar(bar_data, "Model", "Score",
                     "G1: Training Pipeline Comparison",
                     os.path.join(out_dir, "g1_pipeline_overall"))

    # Per-category accuracy
    categories = ["random", "popular", "adversarial"]
    cat_data = {}
    for name, m in data.items():
        cat_data[name] = {cat.capitalize(): m.get(f"{cat}_accuracy", 0) for cat in categories}
    plot_grouped_bar(cat_data, "Model", "Accuracy",
                     "G1: POPE Accuracy by Category",
                     os.path.join(out_dir, "g1_pipeline_by_category"))

    # Yes-ratio
    yr_data = {}
    for name, m in data.items():
        yr_data[name] = {cat.capitalize(): m.get(f"{cat}_yes_ratio", 0) for cat in categories}
    plot_grouped_bar(yr_data, "Model", "Yes Ratio",
                     "G1: Yes-Ratio by Category (ideal=0.5)",
                     os.path.join(out_dir, "g1_pipeline_yes_ratio"))

    _print_table("G1: Training Pipeline", data)
    return data


def analyze_lora_rank(eval_root: str, out_dir: str):
    """G2a: LoRA rank = 4, 8, 16, 32."""
    ranks = [4, 8, 16, 32]
    rank_dirs = {
        4: "ablation_sft_r4",
        8: "sft",
        16: "ablation_sft_r16",
        32: "ablation_sft_r32",
    }
    data = {}
    for r, dirname in rank_dirs.items():
        m = load_pope_metrics(os.path.join(eval_root, dirname))
        if m:
            data[r] = m

    if len(data) < 2:
        print(f"  [SKIP] LoRA rank: only {len(data)} variants found")
        return

    available_ranks = sorted(data.keys())
    metrics_to_plot = {
        "Accuracy": [data[r]["accuracy"] for r in available_ranks],
        "F1": [data[r]["f1"] for r in available_ranks],
        "Precision": [data[r]["precision"] for r in available_ranks],
    }
    plot_line(available_ranks, metrics_to_plot, "LoRA Rank", "Score",
              "G2a: Effect of LoRA Rank on POPE Performance",
              os.path.join(out_dir, "g2a_lora_rank"))

    # Adversarial accuracy specifically
    adv_data = {
        "Adversarial Acc": [data[r].get("adversarial_accuracy", 0) for r in available_ranks],
        "Adversarial F1": [data[r].get("adversarial_f1", 0) for r in available_ranks],
    }
    plot_line(available_ranks, adv_data, "LoRA Rank", "Score",
              "G2a: LoRA Rank vs Adversarial POPE",
              os.path.join(out_dir, "g2a_lora_rank_adversarial"))

    _print_table("G2a: LoRA Rank", {f"r={r}": m for r, m in data.items()})
    return data


def analyze_lora_target(eval_root: str, out_dir: str):
    """G2b: LoRA target = q_proj,v_proj vs all."""
    target_dirs = {
        "q_proj,v_proj": "ablation_target_qv",
        "all": "ablation_target_all",
    }
    data = {}
    for name, dirname in target_dirs.items():
        m = load_pope_metrics(os.path.join(eval_root, dirname))
        if m:
            data[name] = m

    if len(data) < 2:
        print(f"  [SKIP] LoRA target: only {len(data)} variants found")
        return

    bar_data = {name: {"Accuracy": m["accuracy"], "Precision": m["precision"],
                       "Recall": m["recall"], "F1": m["f1"]}
                for name, m in data.items()}
    plot_grouped_bar(bar_data, "LoRA Target", "Score",
                     "G2b: Effect of LoRA Target Modules",
                     os.path.join(out_dir, "g2b_lora_target"))

    _print_table("G2b: LoRA Target", data)
    return data


def analyze_data_scale(eval_root: str, out_dir: str):
    """G3: SFT data scaling = 5K, 10K, 25K, 50K."""
    scales = [5000, 10000, 25000, 50000]
    scale_dirs = {
        5000: "ablation_sft_data5k",
        10000: "ablation_sft_data10k",
        25000: "ablation_sft_data25k",
        50000: "sft",
    }
    data = {}
    for s, dirname in scale_dirs.items():
        m = load_pope_metrics(os.path.join(eval_root, dirname))
        if m:
            data[s] = m

    if len(data) < 2:
        print(f"  [SKIP] Data scale: only {len(data)} variants found")
        return

    available_scales = sorted(data.keys())
    x_labels = [f"{s // 1000}K" for s in available_scales]

    metrics_to_plot = {
        "Accuracy": [data[s]["accuracy"] for s in available_scales],
        "F1": [data[s]["f1"] for s in available_scales],
    }
    plot_line(available_scales, metrics_to_plot, "SFT Training Samples", "Score",
              "G3: Data Scaling Law (SFT)",
              os.path.join(out_dir, "g3_data_scale"),
              x_labels=x_labels)

    # Yes-ratio trend
    yr_data = {
        "Yes Ratio (overall)": [data[s]["yes_ratio"] for s in available_scales],
    }
    plot_line(available_scales, yr_data, "SFT Training Samples", "Yes Ratio",
              "G3: Data Scale vs Yes-Ratio",
              os.path.join(out_dir, "g3_data_scale_yes_ratio"),
              x_labels=x_labels, ideal_line=0.5)

    _print_table("G3: Data Scale", {f"{s // 1000}K": m for s, m in data.items()})
    return data


def analyze_sft_lr(eval_root: str, out_dir: str):
    """G4a: SFT learning rate = 5e-5, 1e-4, 2e-4."""
    lrs = [5e-5, 1e-4, 2e-4]
    lr_dirs = {
        5e-5: "ablation_sft_lr5e5",
        1e-4: "ablation_sft_lr1e4",
        2e-4: "ablation_sft_lr2e4",
    }
    data = {}
    for lr, dirname in lr_dirs.items():
        m = load_pope_metrics(os.path.join(eval_root, dirname))
        if m:
            data[lr] = m

    if len(data) < 2:
        print(f"  [SKIP] SFT LR: only {len(data)} variants found")
        return

    available_lrs = sorted(data.keys())
    x_labels = [f"{lr:.0e}" for lr in available_lrs]

    metrics_to_plot = {
        "Accuracy": [data[lr]["accuracy"] for lr in available_lrs],
        "F1": [data[lr]["f1"] for lr in available_lrs],
    }
    plot_line(available_lrs, metrics_to_plot, "SFT Learning Rate", "Score",
              "G4a: SFT Learning Rate Sensitivity",
              os.path.join(out_dir, "g4a_sft_lr"),
              x_labels=x_labels)

    _print_table("G4a: SFT Learning Rate", {f"lr={lr:.0e}": m for lr, m in data.items()})
    return data


def analyze_sft_epoch(eval_root: str, out_dir: str):
    """G4b: SFT epochs = 1, 2, 3."""
    epochs = [1, 2, 3]
    epoch_dirs = {
        1: "ablation_sft_epoch1",
        2: "ablation_sft_epoch2",
        3: "ablation_sft_epoch3",
    }
    data = {}
    for ep, dirname in epoch_dirs.items():
        m = load_pope_metrics(os.path.join(eval_root, dirname))
        if m:
            data[ep] = m

    if len(data) < 2:
        print(f"  [SKIP] SFT epochs: only {len(data)} variants found")
        return

    available_epochs = sorted(data.keys())
    metrics_to_plot = {
        "Accuracy": [data[e]["accuracy"] for e in available_epochs],
        "F1": [data[e]["f1"] for e in available_epochs],
    }
    plot_line(available_epochs, metrics_to_plot, "SFT Epochs", "Score",
              "G4b: SFT Training Epochs",
              os.path.join(out_dir, "g4b_sft_epoch"))

    _print_table("G4b: SFT Epochs", {f"epoch={e}": m for e, m in data.items()})
    return data


def analyze_dpo_beta(eval_root: str, out_dir: str):
    """G5a: DPO beta = 0.01, 0.05, 0.1, 0.2, 0.5, 1.0."""
    betas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    beta_dirs = {
        0.01: "ablation_dpo_beta001",
        0.05: "ablation_dpo_beta005",
        0.1: "dpo",
        0.2: "ablation_dpo_beta02",
        0.5: "ablation_dpo_beta05",
        1.0: "ablation_dpo_beta10",
    }
    data = {}
    for b, dirname in beta_dirs.items():
        m = load_pope_metrics(os.path.join(eval_root, dirname))
        if m:
            data[b] = m

    if len(data) < 2:
        print(f"  [SKIP] DPO beta: only {len(data)} variants found")
        return

    available_betas = sorted(data.keys())
    x_labels = [str(b) for b in available_betas]

    metrics_to_plot = {
        "Accuracy": [data[b]["accuracy"] for b in available_betas],
        "F1": [data[b]["f1"] for b in available_betas],
        "Precision": [data[b]["precision"] for b in available_betas],
    }
    plot_line(available_betas, metrics_to_plot, "DPO Beta (β)", "Score",
              "G5a: DPO Beta Sensitivity",
              os.path.join(out_dir, "g5a_dpo_beta"),
              x_labels=x_labels)

    # Reward margin proxy: yes_ratio deviation from 0.5
    yr_data = {
        "Yes Ratio": [data[b]["yes_ratio"] for b in available_betas],
    }
    plot_line(available_betas, yr_data, "DPO Beta (β)", "Yes Ratio",
              "G5a: DPO Beta vs Yes-Ratio",
              os.path.join(out_dir, "g5a_dpo_beta_yes_ratio"),
              x_labels=x_labels, ideal_line=0.5)

    _print_table("G5a: DPO Beta", {f"β={b}": m for b, m in data.items()})
    return data


def analyze_dpo_loss(eval_root: str, out_dir: str):
    """G5b: DPO loss = sigmoid, hinge, ipo."""
    loss_dirs = {
        "Sigmoid": "dpo",
        "Hinge": "ablation_dpo_hinge",
        "IPO": "ablation_dpo_ipo",
    }
    data = {}
    for name, dirname in loss_dirs.items():
        m = load_pope_metrics(os.path.join(eval_root, dirname))
        if m:
            data[name] = m

    if len(data) < 2:
        print(f"  [SKIP] DPO loss: only {len(data)} variants found")
        return

    bar_data = {name: {"Accuracy": m["accuracy"], "Precision": m["precision"],
                       "Recall": m["recall"], "F1": m["f1"]}
                for name, m in data.items()}
    plot_grouped_bar(bar_data, "DPO Loss Function", "Score",
                     "G5b: DPO Loss Function Comparison",
                     os.path.join(out_dir, "g5b_dpo_loss"))

    _print_table("G5b: DPO Loss", data)
    return data


def analyze_dpo_lr(eval_root: str, out_dir: str):
    """G5c: DPO learning rate = 1e-6, 5e-6, 1e-5."""
    lrs = [1e-6, 5e-6, 1e-5]
    lr_dirs = {
        1e-6: "ablation_dpo_lr1e6",
        5e-6: "ablation_dpo_lr5e6",
        1e-5: "ablation_dpo_lr1e5",
    }
    data = {}
    for lr, dirname in lr_dirs.items():
        m = load_pope_metrics(os.path.join(eval_root, dirname))
        if m:
            data[lr] = m

    if len(data) < 2:
        print(f"  [SKIP] DPO LR: only {len(data)} variants found")
        return

    available_lrs = sorted(data.keys())
    x_labels = [f"{lr:.0e}" for lr in available_lrs]

    metrics_to_plot = {
        "Accuracy": [data[lr]["accuracy"] for lr in available_lrs],
        "F1": [data[lr]["f1"] for lr in available_lrs],
    }
    plot_line(available_lrs, metrics_to_plot, "DPO Learning Rate", "Score",
              "G5c: DPO Learning Rate Sensitivity",
              os.path.join(out_dir, "g5c_dpo_lr"),
              x_labels=x_labels)

    _print_table("G5c: DPO Learning Rate", {f"lr={lr:.0e}": m for lr, m in data.items()})
    return data


def analyze_resolution(eval_root: str, out_dir: str):
    """G6: Image resolution = 128², 256², 512²."""
    res_dirs = {
        "128² (16K px)": "ablation_lowres",
        "256² (65K px)": "ablation_midres",
        "512² (262K px)": "ablation_highres",
    }
    data = {}
    for name, dirname in res_dirs.items():
        m = load_pope_metrics(os.path.join(eval_root, dirname))
        if m:
            data[name] = m

    if len(data) < 2:
        print(f"  [SKIP] Resolution: only {len(data)} variants found")
        return

    bar_data = {name: {"Accuracy": m["accuracy"], "Precision": m["precision"],
                       "Recall": m["recall"], "F1": m["f1"]}
                for name, m in data.items()}
    plot_grouped_bar(bar_data, "Image Resolution", "Score",
                     "G6: Effect of Image Resolution on Hallucination",
                     os.path.join(out_dir, "g6_resolution"))

    # Per-category accuracy
    categories = ["random", "popular", "adversarial"]
    cat_data = {}
    for name, m in data.items():
        cat_data[name] = {cat.capitalize(): m.get(f"{cat}_accuracy", 0) for cat in categories}
    plot_grouped_bar(cat_data, "Resolution", "Accuracy",
                     "G6: Resolution vs POPE Accuracy by Category",
                     os.path.join(out_dir, "g6_resolution_by_category"))

    _print_table("G6: Image Resolution", data)
    return data


# ============================================================
# Summary Table
# ============================================================

def _print_table(title: str, data: dict[str, dict]):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    header = f"{'Variant':<22} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Yes%':<8}"
    print(header)
    print("-" * len(header))
    for name, m in data.items():
        print(f"{name:<22} {m['accuracy']:.4f}  {m['precision']:.4f}  "
              f"{m['recall']:.4f}  {m['f1']:.4f}  {m['yes_ratio']:.4f}")
    print()


def generate_summary_table(eval_root: str, out_dir: str):
    """Generate a comprehensive LaTeX-ready summary of all ablation results."""
    summary = {}

    # Scan all eval directories
    eval_root = Path(eval_root)
    for d in sorted(eval_root.iterdir()):
        if d.is_dir():
            m = load_pope_metrics(str(d))
            if m:
                summary[d.name] = m

    if not summary:
        print("No evaluation results found.")
        return

    # Save as JSON
    out_path = os.path.join(out_dir, "ablation_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {out_path}")

    # Save as CSV for easy import
    csv_path = os.path.join(out_dir, "ablation_summary.csv")
    with open(csv_path, "w") as f:
        headers = ["experiment", "accuracy", "precision", "recall", "f1", "yes_ratio",
                    "random_acc", "popular_acc", "adversarial_acc"]
        f.write(",".join(headers) + "\n")
        for name, m in summary.items():
            row = [name, m["accuracy"], m["precision"], m["recall"], m["f1"], m["yes_ratio"],
                   m.get("random_accuracy", ""), m.get("popular_accuracy", ""),
                   m.get("adversarial_accuracy", "")]
            f.write(",".join(str(v) for v in row) + "\n")
    print(f"CSV summary saved to {csv_path}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation experiment analysis")
    parser.add_argument("--eval_root", type=str, default="results/eval",
                        help="Root directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default="results/figures/ablation",
                        help="Output directory for charts")
    parser.add_argument("--group", type=str, default="all",
                        choices=["all", "pipeline", "lora_rank", "lora_target",
                                 "data_scale", "sft_lr", "sft_epoch",
                                 "dpo_beta", "dpo_loss", "dpo_lr", "resolution",
                                 "summary"],
                        help="Which ablation group to analyze")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    analyzers = {
        "pipeline": analyze_pipeline,
        "lora_rank": analyze_lora_rank,
        "lora_target": analyze_lora_target,
        "data_scale": analyze_data_scale,
        "sft_lr": analyze_sft_lr,
        "sft_epoch": analyze_sft_epoch,
        "dpo_beta": analyze_dpo_beta,
        "dpo_loss": analyze_dpo_loss,
        "dpo_lr": analyze_dpo_lr,
        "resolution": analyze_resolution,
    }

    if args.group == "all":
        for name, func in analyzers.items():
            func(args.eval_root, args.output_dir)
        generate_summary_table(args.eval_root, args.output_dir)
    elif args.group == "summary":
        generate_summary_table(args.eval_root, args.output_dir)
    else:
        analyzers[args.group](args.eval_root, args.output_dir)

    print(f"\nAll charts saved to {args.output_dir}/")
