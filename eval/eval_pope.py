"""
Evaluate POPE results: accuracy, precision, recall, F1, yes-ratio.
Supports per-category breakdown (random, popular, adversarial).
"""

import argparse
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_yesno(text: str) -> str:
    import re
    text = text.strip()
    # Strip all <think>/<\/think> tags (Qwen3-VL thinking mode)
    # DPO models may output malformed tags like <think>...<think> (double open, no close)
    text = re.sub(r"</?think>", "", text).strip().lower()
    if text.startswith("yes"):
        return "yes"
    return "no"


def compute_metrics(results: list[dict]) -> dict:
    gt = [1 if r["gt_answer"] == "yes" else 0 for r in results]
    pred = [1 if parse_yesno(r["answer"]) == "yes" else 0 for r in results]

    return {
        "accuracy": accuracy_score(gt, pred),
        "precision": precision_score(gt, pred, zero_division=0),
        "recall": recall_score(gt, pred, zero_division=0),
        "f1": f1_score(gt, pred, zero_division=0),
        "yes_ratio": sum(pred) / len(pred) if pred else 0,
        "total": len(results),
    }


def evaluate_pope(result_file: str) -> dict:
    with open(result_file, "r") as f:
        results = json.load(f)

    overall = compute_metrics(results)

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_metrics = compute_metrics(cat_results)
        for k, v in cat_metrics.items():
            overall[f"{cat}_{k}"] = v

    return overall


def print_metrics(metrics: dict, label: str = ""):
    if label:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"{'=' * 60}")

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Yes Ratio: {metrics['yes_ratio']:.4f}")
    print(f"  Total:     {metrics['total']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", type=str, default=None, help="Single result file")
    parser.add_argument("--input_dir", type=str, default=None, help="Directory with pope_*.json files")
    parser.add_argument("--output_dir", type=str, default=None, help="Save metrics JSON to this dir")
    parser.add_argument("--output_json", type=str, default=None, help="Save metrics to single JSON file")
    args = parser.parse_args()

    import os

    if args.input_dir:
        # Batch mode: evaluate all pope_*.json in directory
        all_results = []
        for fname in sorted(os.listdir(args.input_dir)):
            if fname.startswith("pope_") and fname.endswith(".json"):
                fpath = os.path.join(args.input_dir, fname)
                metrics = evaluate_pope(fpath)
                print_metrics(metrics, label=fname)
                all_results.append((fname, metrics))

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            combined = {}
            for fname, metrics in all_results:
                combined[fname] = metrics
            out_path = os.path.join(args.output_dir, "pope_metrics.json")
            with open(out_path, "w") as f:
                json.dump(combined, f, indent=2)
            print(f"\nAll metrics saved to {out_path}")

    elif args.result_file:
        metrics = evaluate_pope(args.result_file)
        print_metrics(metrics, label=args.result_file)

        if args.output_json:
            with open(args.output_json, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\nMetrics saved to {args.output_json}")
    else:
        parser.error("Either --result_file or --input_dir is required")
