"""
Evaluate MMBench benchmark results.

Scoring:
- Vanilla: accuracy = correct / total (per category and overall)
- CircularEval: correct only if ALL 4 permutations agree on the right answer

Usage:
    python eval/eval_mmbench.py \
        --input_file results/eval/base/mmbench_answers.json \
        --output_file results/eval/base/mmbench_metrics.json
"""

import argparse
import json
import os


def evaluate_vanilla(results: list[dict]) -> dict:
    """Standard accuracy evaluation."""
    by_category = {}
    by_l2 = {}

    for item in results:
        cat = item.get("category", "unknown")
        l2 = item.get("l2_category", "")
        pred = item.get("prediction", "unknown")
        gt = item.get("gt_answer", "")

        correct = pred.upper() == gt.upper()

        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        by_category[cat]["total"] += 1
        if correct:
            by_category[cat]["correct"] += 1

        if l2:
            if l2 not in by_l2:
                by_l2[l2] = {"correct": 0, "total": 0}
            by_l2[l2]["total"] += 1
            if correct:
                by_l2[l2]["correct"] += 1

    # Compute accuracies
    metrics = {}
    total_correct = 0
    total_count = 0

    for cat, data in sorted(by_category.items()):
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        metrics[cat] = {
            "accuracy": round(acc, 2),
            "correct": data["correct"],
            "total": data["total"],
        }
        total_correct += data["correct"]
        total_count += data["total"]

    # L2 category breakdown
    l2_metrics = {}
    for l2, data in sorted(by_l2.items()):
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        l2_metrics[l2] = {
            "accuracy": round(acc, 2),
            "correct": data["correct"],
            "total": data["total"],
        }

    overall_acc = total_correct / total_count * 100 if total_count > 0 else 0
    metrics["_summary"] = {
        "overall_accuracy": round(overall_acc, 2),
        "total_correct": total_correct,
        "total_questions": total_count,
        "categories": len(by_category),
        "circular_eval": False,
    }
    metrics["_l2_categories"] = l2_metrics

    return metrics


def evaluate_circular(results: list[dict]) -> dict:
    """CircularEval: correct only if ALL permutations give the right answer."""
    # Group by question_id
    by_question = {}
    for item in results:
        qid = item["question_id"]
        if qid not in by_question:
            by_question[qid] = {
                "gt_answer": item.get("gt_answer", ""),
                "category": item.get("category", "unknown"),
                "l2_category": item.get("l2_category", ""),
                "predictions": [],
            }
        by_question[qid]["predictions"].append(item.get("prediction", "unknown"))

    by_category = {}
    by_l2 = {}

    for qid, qdata in by_question.items():
        cat = qdata["category"]
        l2 = qdata["l2_category"]
        gt = qdata["gt_answer"].upper()

        # CircularEval: ALL permutations must be correct
        all_correct = all(p.upper() == gt for p in qdata["predictions"])

        if cat not in by_category:
            by_category[cat] = {"correct": 0, "total": 0}
        by_category[cat]["total"] += 1
        if all_correct:
            by_category[cat]["correct"] += 1

        if l2:
            if l2 not in by_l2:
                by_l2[l2] = {"correct": 0, "total": 0}
            by_l2[l2]["total"] += 1
            if all_correct:
                by_l2[l2]["correct"] += 1

    metrics = {}
    total_correct = 0
    total_count = 0

    for cat, data in sorted(by_category.items()):
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        metrics[cat] = {
            "accuracy": round(acc, 2),
            "correct": data["correct"],
            "total": data["total"],
        }
        total_correct += data["correct"]
        total_count += data["total"]

    l2_metrics = {}
    for l2, data in sorted(by_l2.items()):
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        l2_metrics[l2] = {
            "accuracy": round(acc, 2),
            "correct": data["correct"],
            "total": data["total"],
        }

    overall_acc = total_correct / total_count * 100 if total_count > 0 else 0
    metrics["_summary"] = {
        "overall_accuracy": round(overall_acc, 2),
        "total_correct": total_correct,
        "total_questions": total_count,
        "categories": len(by_category),
        "circular_eval": True,
        "permutations_per_question": len(next(iter(by_question.values()))["predictions"]) if by_question else 0,
    }
    metrics["_l2_categories"] = l2_metrics

    return metrics


def print_metrics(metrics: dict):
    """Pretty-print MMBench metrics."""
    summary = metrics["_summary"]
    circular = summary.get("circular_eval", False)

    print("=" * 60)
    print(f"  MMBench Evaluation Results {'(CircularEval)' if circular else '(Vanilla)'}")
    print("=" * 60)

    # Per-category results
    print(f"\n  {'Category':<30s} {'Accuracy':>8s} {'Correct':>8s} {'Total':>6s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*6}")

    for key, val in sorted(metrics.items()):
        if key.startswith("_"):
            continue
        print(f"  {key:<30s} {val['accuracy']:>7.2f}% {val['correct']:>8d} {val['total']:>6d}")

    print(f"\n  {'OVERALL':<30s} {summary['overall_accuracy']:>7.2f}% "
          f"{summary['total_correct']:>8d} {summary['total_questions']:>6d}")
    print(f"  Categories: {summary['categories']}")

    # L2 categories (brief)
    l2 = metrics.get("_l2_categories", {})
    if l2:
        print(f"\n  L2 Sub-categories ({len(l2)}):")
        for key, val in sorted(l2.items(), key=lambda x: -x[1]["accuracy"]):
            print(f"    {key:<35s} {val['accuracy']:>7.2f}% ({val['correct']}/{val['total']})")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to mmbench_answers.json")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--circular", action="store_true",
                        help="Use CircularEval scoring (requires circular_eval data)")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    # Auto-detect circular eval data
    has_permutations = any("permutation" in r for r in results)

    if args.circular and not has_permutations:
        print("Warning: --circular requested but data has no permutation field. "
              "Falling back to vanilla evaluation.")
        args.circular = False

    if args.circular and has_permutations:
        metrics = evaluate_circular(results)
    else:
        # For circular data without --circular flag, use only permutation 0
        if has_permutations:
            results = [r for r in results if r.get("permutation", 0) == 0]
            print(f"Using permutation 0 only: {len(results)} questions")
        metrics = evaluate_vanilla(results)

    print_metrics(metrics)

    if args.output_file:
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to {args.output_file}")


if __name__ == "__main__":
    main()
