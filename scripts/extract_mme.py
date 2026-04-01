"""Extract MME dataset from HuggingFace Parquet to evaluation format on server."""
import json
import os
from datasets import load_dataset

PERCEPTION = [
    "existence", "count", "position", "color", "posters",
    "celebrity", "scene", "landmark", "artwork", "OCR"
]

ds = load_dataset("parquet", data_files="data/mme_raw/data/*.parquet", split="train")
print(f"Loaded {len(ds)} items, columns: {ds.column_names}")

# Inspect question_id format
seen_cats = set()
for item in ds:
    cat = item["category"]
    if cat not in seen_cats:
        seen_cats.add(cat)
        print(f"  {cat}: qid={item['question_id']}, answer={item['answer']}")

out_dir = "data/mme"
os.makedirs(out_dir, exist_ok=True)

questions = []
saved_images = set()

for idx, item in enumerate(ds):
    cat = item["category"]
    qid = str(item["question_id"])

    cat_dir = os.path.join(out_dir, cat)
    os.makedirs(cat_dir, exist_ok=True)

    # question_id format: "category/filename.ext" (e.g., "code_reasoning/0020.png")
    # Extract just the filename part
    if "/" in qid:
        img_file = qid.split("/", 1)[1]  # "0020.png"
    else:
        img_file = qid

    # Ensure valid image extension
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_file = img_file + ".jpg"

    img_id = os.path.splitext(img_file)[0]
    q_idx = idx  # use global index as unique id

    img_path = os.path.join(cat_dir, img_file)

    # Save image once
    img_key = f"{cat}/{img_file}"
    if img_key not in saved_images:
        item["image"].save(img_path)
        saved_images.add(img_key)

    # Write txt file (for official format compatibility)
    txt_name = os.path.splitext(img_file)[0] + ".txt"
    txt_path = os.path.join(cat_dir, txt_name)
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"{item['question']}\t{item['answer']}\n")

    split = "perception" if cat in PERCEPTION else "cognition"
    questions.append({
        "question_id": qid,
        "image": img_path,
        "image_file": img_file,
        "question": item["question"],
        "answer": item["answer"],
        "category": cat,
        "split": split,
    })

    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx + 1}/{len(ds)}")

# Save questions.json
json_path = os.path.join(out_dir, "questions.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(questions, f, ensure_ascii=False, indent=2)

print(f"\nDone: {len(questions)} questions, {len(saved_images)} images")
print(f"Categories: {sorted(seen_cats)}")

# Verify: count questions per category
from collections import Counter
cat_counts = Counter(q["category"] for q in questions)
for cat, cnt in sorted(cat_counts.items()):
    print(f"  {cat}: {cnt} questions")
