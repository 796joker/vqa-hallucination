"""
Evaluate MME benchmark results.

Scoring logic (official):
- For each question: correct if parsed answer matches ground truth
- acc = correct / total per subtask
- acc_plus = both paired questions correct / total image pairs per subtask
- subtask_score = acc * N + acc_plus * N (where N = number of images)
- Perception = sum of 10 perception subtask scores (max 2000)
- Cognition = sum of 4 cognition subtask scores (max 800)

Usage:
    python eval/eval_mme.py \
        --input_file results/eval/base/mme_answers.json \
        --output_file results/eval/base/mme_metrics.json
"""

import argparse
import json
import os
import re

PERCEPTION_TASKS = [
    "existence", "count", "position", "color", "posters",
    "celebrity", "scene", "landmark", "artwork", "OCR"
]
COGNITION_TASKS = [
    "commonsense_reasoning", "numerical_calculation",
    "text_translation", "code_reasoning"
]


def parse_yes_no(text: str) -> str:
    """Parse model output to yes/no/other."""
    text = re.sub(r"</?think>", "", text)
    text = text.strip().lower().rstrip(".")

    if text.startswith("yes"):
        return "yes"
    elif text.startswith("no"):
        return "no"
    else:
        return "other"


def evaluate_mme(results: list[dict]) -> dict:
    """Compute MME metrics from model predictions."""

    # Group by category
    by_category = {}
    for item in results:
        cat = item["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(item)

    metrics = {}
    perception_total = 0.0
    cognition_total = 0.0

    all_tasks = PERCEPTION_TASKS + COGNITION_TASKS
    for task in all_tasks:
        if task not in by_category:
            continue

        items = by_category[task]
        split = "perception" if task in PERCEPTION_TASKS else "cognition"

        # Compute accuracy
        correct = 0
        total = 0
        for item in items:
            pred = parse_yes_no(item["prediction"])
            gt = item["gt_answer"].strip().lower()
            if pred == gt:
                correct += 1
            total += 1

        acc = correct / total if total > 0 else 0

        # Compute acc_plus (both paired questions for same image must be correct)
        # Group by image
        by_image = {}
        for item in items:
            img = item["image"]
            if img not in by_image:
                by_image[img] = []
            by_image[img].append(item)

        pair_correct = 0
        pair_total = 0
        for img, img_items in by_image.items():
            all_correct = True
            for item in img_items:
                pred = parse_yes_no(item["prediction"])
                gt = item["gt_answer"].strip().lower()
                if pred != gt:
                    all_correct = False
                    break
            if all_correct:
                pair_correct += 1
            pair_total += 1

        acc_plus = pair_correct / pair_total if pair_total > 0 else 0

        # Score = acc * N_images + acc_plus * N_images (max = 2 * N_images)
        n_images = pair_total
        score = (acc + acc_plus) * n_images

        metrics[task] = {
            "accuracy": round(acc * 100, 2),
            "accuracy_plus": round(acc_plus * 100, 2),
            "score": round(score, 2),
            "max_score": n_images * 2,
            "total_questions": total,
            "total_images": n_images,
            "correct": correct,
            "pair_correct": pair_correct,
            "split": split,
        }

        if split == "perception":
            perception_total += score
        else:
            cognition_total += score

    # Summary
    metrics["_summary"] = {
        "perception_score": round(perception_total, 2),
        "cognition_score": round(cognition_total, 2),
        "total_score": round(perception_total + cognition_total, 2),
        "perception_max": 2000,
        "cognition_max": 800,
        "total_max": 2800,
        "total_questions": len(results),
        "total_categories": len(by_category),
    }

    return metrics


def print_metrics(metrics: dict):
    """Pretty-print MME metrics."""
    summary = metrics["_summary"]

    print("=" * 60)
    print("  MME Evaluation Results")
    print("=" * 60)

    # Perception tasks
    print("\n  [Perception]")
    for task in PERCEPTION_TASKS:
        if task in metrics:
            m = metrics[task]
            print(f"    {task:25s}  acc={m['accuracy']:6.2f}%  acc+={m['accuracy_plus']:6.2f}%  score={m['score']:7.2f}/{m['max_score']}")

    print(f"\n    {'PERCEPTION TOTAL':25s}  {summary['perception_score']:7.2f} / {summary['perception_max']}")

    # Cognition tasks
    print("\n  [Cognition]")
    for task in COGNITION_TASKS:
        if task in metrics:
            m = metrics[task]
            print(f"    {task:25s}  acc={m['accuracy']:6.2f}%  acc+={m['accuracy_plus']:6.2f}%  score={m['score']:7.2f}/{m['max_score']}")

    print(f"\n    {'COGNITION TOTAL':25s}  {summary['cognition_score']:7.2f} / {summary['cognition_max']}")

    print(f"\n    {'TOTAL':25s}  {summary['total_score']:7.2f} / {summary['total_max']}")
    print(f"    Questions: {summary['total_questions']}, Categories: {summary['total_categories']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to mme_answers.json")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing mme_answers.json (alternative)")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    # Determine input file
    if args.input_dir:
        input_file = os.path.join(args.input_dir, "mme_answers.json")
    else:
        input_file = args.input_file

    with open(input_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    metrics = evaluate_mme(results)
    print_metrics(metrics)

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to {args.output_file}")


if __name__ == "__main__":
    main()
