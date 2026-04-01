"""
Compare training curves across ablation experiments.
Reads trainer_state.json from each experiment and overlays loss/metrics curves.

Usage:
    python eval/compare_training_curves.py --output_dir results/figures/training
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 11


def load_training_log(result_dir: str) -> list[dict] | None:
    """Load training history from trainer_state.json."""
    state_path = os.path.join(result_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        return None

    with open(state_path) as f:
        state = json.load(f)

    return state.get("log_history", [])


def extract_metric(log_history: list[dict], metric: str) -> tuple[list, list]:
    """Extract (steps, values) for a given metric from log history."""
    steps, values = [], []
    for entry in log_history:
        if metric in entry:
            steps.append(entry.get("step", 0))
            values.append(entry[metric])
    return steps, values


def save_fig(fig, path_base: str):
    fig.savefig(f"{path_base}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(f"{path_base}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974",
          "#64B5CD", "#D65F5F", "#4878CF", "#6ACC65", "#D5BB67"]


def compare_curves(experiments: dict[str, str], metric: str, title: str,
                   ylabel: str, output_path: str):
    """Plot training curves for multiple experiments on the same axes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    plotted = 0
    for i, (name, result_dir) in enumerate(experiments.items()):
        log = load_training_log(result_dir)
        if log is None:
            print(f"  [SKIP] {name}: no trainer_state.json found")
            continue

        steps, values = extract_metric(log, metric)
        if not values:
            continue

        color = COLORS[i % len(COLORS)]
        ax.plot(steps, values, color=color, label=name, linewidth=1.5, alpha=0.85)
        plotted += 1

    if plotted < 2:
        print(f"  [SKIP] Not enough data to compare for metric '{metric}'")
        plt.close(fig)
        return

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(fig, output_path)
    print(f"  Saved: {output_path}")


def compare_lora_rank(output_dir: str):
    """Compare SFT training curves across LoRA ranks."""
    experiments = {
        "r=4": "results/ablation/sft_r4",
        "r=8 (baseline)": "results/sft/lora_r8",
        "r=16": "results/ablation/sft_lora_r16",
        "r=32": "results/ablation/sft_lora_r32",
    }

    print("\n--- LoRA Rank: Training Loss ---")
    compare_curves(experiments, "loss", "SFT Training Loss by LoRA Rank",
                   "Loss", os.path.join(output_dir, "train_loss_lora_rank"))

    print("\n--- LoRA Rank: Eval Loss ---")
    compare_curves(experiments, "eval_loss", "SFT Eval Loss by LoRA Rank",
                   "Eval Loss", os.path.join(output_dir, "eval_loss_lora_rank"))


def compare_data_scale(output_dir: str):
    """Compare SFT training curves across data scales."""
    experiments = {
        "5K": "results/ablation/sft_data5k",
        "10K": "results/ablation/sft_data10k",
        "25K": "results/ablation/sft_data25k",
        "50K (baseline)": "results/sft/lora_r8",
    }

    print("\n--- Data Scale: Training Loss ---")
    compare_curves(experiments, "loss", "SFT Training Loss by Data Scale",
                   "Loss", os.path.join(output_dir, "train_loss_data_scale"))


def compare_dpo_beta(output_dir: str):
    """Compare DPO training curves across beta values."""
    experiments = {
        "β=0.01": "results/ablation/dpo_beta001",
        "β=0.05": "results/ablation/dpo_beta005",
        "β=0.1 (baseline)": "results/dpo/lora_r8_beta01",
        "β=0.2": "results/ablation/dpo_beta02",
        "β=0.5": "results/ablation/dpo_beta05",
        "β=1.0": "results/ablation/dpo_beta10",
    }

    print("\n--- DPO Beta: Training Loss ---")
    compare_curves(experiments, "loss", "DPO Training Loss by Beta",
                   "Loss", os.path.join(output_dir, "train_loss_dpo_beta"))

    print("\n--- DPO Beta: Reward Accuracy ---")
    compare_curves(experiments, "rewards/accuracies",
                   "DPO Reward Accuracy by Beta",
                   "Reward Accuracy",
                   os.path.join(output_dir, "reward_acc_dpo_beta"))

    print("\n--- DPO Beta: Reward Margin ---")
    compare_curves(experiments, "rewards/margins",
                   "DPO Reward Margin by Beta",
                   "Reward Margin",
                   os.path.join(output_dir, "reward_margin_dpo_beta"))


def compare_dpo_loss(output_dir: str):
    """Compare DPO training curves across loss functions."""
    experiments = {
        "Sigmoid (DPO)": "results/dpo/lora_r8_beta01",
        "Hinge": "results/ablation/dpo_loss_hinge",
        "IPO": "results/ablation/dpo_loss_ipo",
    }

    print("\n--- DPO Loss: Training Loss ---")
    compare_curves(experiments, "loss", "DPO Training Loss by Loss Function",
                   "Loss", os.path.join(output_dir, "train_loss_dpo_loss"))

    print("\n--- DPO Loss: Reward Accuracy ---")
    compare_curves(experiments, "rewards/accuracies",
                   "DPO Reward Accuracy by Loss Function",
                   "Reward Accuracy",
                   os.path.join(output_dir, "reward_acc_dpo_loss"))


def compare_sft_baseline_vs_dpo(output_dir: str):
    """Compare SFT and DPO baseline training dynamics."""
    # SFT loss
    sft_log = load_training_log("results/sft/lora_r8")
    dpo_log = load_training_log("results/dpo/lora_r8_beta01")

    if sft_log and dpo_log:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # SFT loss
        steps, values = extract_metric(sft_log, "loss")
        ax1.plot(steps, values, color=COLORS[0], linewidth=1.5)
        ax1.set_title("SFT Training Loss", fontsize=13)
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("Loss")
        ax1.grid(alpha=0.3)

        # DPO loss + reward accuracy
        steps, values = extract_metric(dpo_log, "loss")
        ax2.plot(steps, values, color=COLORS[2], label="Loss", linewidth=1.5)
        ax2_twin = ax2.twinx()
        steps2, values2 = extract_metric(dpo_log, "rewards/accuracies")
        ax2_twin.plot(steps2, values2, color=COLORS[1], label="Reward Acc", linewidth=1.5, linestyle="--")
        ax2.set_title("DPO Training Dynamics", fontsize=13)
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Loss", color=COLORS[2])
        ax2_twin.set_ylabel("Reward Accuracy", color=COLORS[1])
        ax2.grid(alpha=0.3)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

        plt.tight_layout()
        save_fig(fig, os.path.join(output_dir, "baseline_training_dynamics"))
        print(f"\n  Saved: baseline_training_dynamics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results/figures/training",
                        help="Output directory for training curve plots")
    parser.add_argument("--group", type=str, default="all",
                        choices=["all", "baseline", "lora_rank", "data_scale",
                                 "dpo_beta", "dpo_loss"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    groups = {
        "baseline": compare_sft_baseline_vs_dpo,
        "lora_rank": compare_lora_rank,
        "data_scale": compare_data_scale,
        "dpo_beta": compare_dpo_beta,
        "dpo_loss": compare_dpo_loss,
    }

    if args.group == "all":
        for name, func in groups.items():
            print(f"\n{'=' * 50}")
            print(f"  Comparing: {name}")
            print(f"{'=' * 50}")
            func(args.output_dir)
    else:
        groups[args.group](args.output_dir)

    print(f"\nAll training curve plots saved to {args.output_dir}/")
