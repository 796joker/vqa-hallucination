"""Verify MME dataset integrity before running evaluation."""
import json
import os
from collections import Counter

with open("data/mme/questions.json") as f:
    questions = json.load(f)

print(f"Total questions: {len(questions)}")

# Check all images exist
missing = sum(1 for q in questions if not os.path.exists(q["image"]))
print(f"Missing images: {missing}")

# Verify paired structure (each image should have exactly 2 questions)
img_counts = Counter(q["image"] for q in questions)
not_paired = {img: cnt for img, cnt in img_counts.items() if cnt != 2}
print(f"Images with != 2 questions: {len(not_paired)}")
for img, cnt in list(not_paired.items())[:3]:
    print(f"  {img}: {cnt} questions")

# Answer distribution (should be 50/50 yes/no)
ans_counts = Counter(q["answer"].strip().lower() for q in questions)
print(f"Answer distribution: {dict(ans_counts)}")

# Check if question already contains "yes or no" prompt
has_prompt = sum(1 for q in questions if "yes or no" in q["question"].lower())
print(f"Questions with built-in yes/no prompt: {has_prompt}/{len(questions)}")

# Perception vs cognition
split_counts = Counter(q["split"] for q in questions)
print(f"Split: {dict(split_counts)}")

# Per-category counts
cat_counts = Counter(q["category"] for q in questions)
print(f"\nPer-category:")
for cat, cnt in sorted(cat_counts.items()):
    print(f"  {cat}: {cnt}")

# Samples
print("\nSamples:")
for cat in ["existence", "celebrity", "commonsense_reasoning"]:
    samples = [q for q in questions if q["category"] == cat][:2]
    for s in samples:
        q_text = s["question"][:90]
        print(f"  [{cat}] Q: {q_text}  A: {s['answer']}")
