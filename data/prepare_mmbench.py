"""
Prepare MMBench evaluation data from parquet files.

Converts MMBench parquet (HuggingFace lmms-lab/MMBench format) to:
- images/ directory with extracted JPEG files
- questions.json with standardized format for eval pipeline

Usage:
    python data/prepare_mmbench.py \
        --input_dir /mnt/disk2/lijunlin/downloads/datasets/MMBench/ \
        --output_dir data/mmbench \
        --split en
"""

import argparse
import base64
import glob
import io
import json
import os

import pandas as pd
from PIL import Image
from tqdm import tqdm


def decode_image(image_data) -> Image.Image | None:
    """Decode image from various formats (dict with bytes, base64 string, raw bytes)."""
    try:
        if isinstance(image_data, dict):
            # HuggingFace datasets format: {"bytes": b"...", "path": None}
            image_bytes = image_data.get("bytes")
            if image_bytes is None:
                return None
        elif isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
        elif isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            return None
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print(f"Warning: failed to decode image: {e}")
        return None


def load_parquets(input_dir: str, split: str) -> pd.DataFrame:
    """Load MMBench parquet files for a given split."""
    # Try common naming patterns
    patterns = [
        os.path.join(input_dir, f"*{split}*.parquet"),
        os.path.join(input_dir, f"*{split}*.tsv"),
        os.path.join(input_dir, split, "*.parquet"),
        os.path.join(input_dir, "*.parquet"),
    ]

    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            print(f"Found {len(files)} files matching: {pattern}")
            break
    else:
        # If no parquet/tsv found, try listing directory for user
        print(f"No parquet/tsv files found. Contents of {input_dir}:")
        if os.path.isdir(input_dir):
            for f in os.listdir(input_dir):
                print(f"  {f}")
        raise FileNotFoundError(f"No data files found in {input_dir} for split '{split}'")

    dfs = []
    for f in files:
        if f.endswith(".parquet"):
            dfs.append(pd.read_parquet(f))
        elif f.endswith(".tsv"):
            dfs.append(pd.read_csv(f, sep="\t"))
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def prepare_mmbench(input_dir: str, output_dir: str, split: str):
    df = load_parquets(input_dir, split)

    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Detect column names (MMBench has varying schemas)
    # Common: index, image, question, hint, A, B, C, D, answer, category, l2-category
    idx_col = "index" if "index" in df.columns else df.columns[0]
    img_col = "image" if "image" in df.columns else None
    q_col = "question" if "question" in df.columns else None
    hint_col = "hint" if "hint" in df.columns else None
    answer_col = "answer" if "answer" in df.columns else None
    cat_col = "category" if "category" in df.columns else None
    l2_col = "L2-category" if "L2-category" in df.columns else (
        "l2-category" if "l2-category" in df.columns else (
            "l2_category" if "l2_category" in df.columns else None
        )
    )

    if q_col is None:
        raise ValueError(f"Cannot find 'question' column. Available: {list(df.columns)}")

    # Check if we have options columns
    has_options = all(c in df.columns for c in ["A", "B", "C", "D"])
    if not has_options:
        # Try lowercase
        has_options = all(c in df.columns for c in ["a", "b", "c", "d"])
        if has_options:
            df.rename(columns={"a": "A", "b": "B", "c": "C", "d": "D"}, inplace=True)

    questions = []
    skipped = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing MMBench"):
        qid = f"mmbench_{row[idx_col]}" if idx_col else f"mmbench_{i}"
        img_filename = f"{i:06d}.jpg"
        img_path = os.path.join(images_dir, img_filename)

        # Extract and save image
        if img_col and pd.notna(row.get(img_col)):
            if not os.path.exists(img_path):
                img = decode_image(row[img_col])
                if img is None:
                    skipped += 1
                    continue
                img.save(img_path, "JPEG", quality=95)
        else:
            skipped += 1
            continue

        # Build question entry
        entry = {
            "question_id": qid,
            "image": os.path.abspath(img_path),
            "image_file": img_filename,
            "question": str(row[q_col]),
        }

        if hint_col and pd.notna(row.get(hint_col)):
            hint_val = str(row[hint_col]).strip()
            entry["hint"] = hint_val if hint_val.lower() != "nan" else ""
        else:
            entry["hint"] = ""

        if has_options:
            options = {}
            for letter in ["A", "B", "C", "D"]:
                val = str(row[letter]) if pd.notna(row.get(letter)) else ""
                if val and val.lower() != "nan":
                    options[letter] = val
            entry["options"] = options

        if answer_col and pd.notna(row.get(answer_col)):
            entry["answer"] = str(row[answer_col]).strip()
        else:
            entry["answer"] = ""

        if cat_col and pd.notna(row.get(cat_col)):
            entry["category"] = str(row[cat_col])
        else:
            entry["category"] = "unknown"

        if l2_col and pd.notna(row.get(l2_col)):
            entry["l2_category"] = str(row[l2_col])
        else:
            entry["l2_category"] = ""

        questions.append(entry)

    # Save questions JSON
    out_path = os.path.join(output_dir, "questions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {len(questions)} questions saved to {out_path}")
    print(f"Images saved to {images_dir}")
    if skipped > 0:
        print(f"Skipped {skipped} entries (missing/invalid images)")

    # Print category distribution
    if questions:
        cats = {}
        for q in questions:
            c = q["category"]
            cats[c] = cats.get(c, 0) + 1
        print(f"\nCategory distribution ({len(cats)} categories):")
        for c, n in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  {c}: {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MMBench data for evaluation")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing MMBench parquet/tsv files")
    parser.add_argument("--output_dir", type=str, default="data/mmbench",
                        help="Output directory for processed data")
    parser.add_argument("--split", type=str, default="en",
                        choices=["en", "cn", "cc", "dev", "test"],
                        help="MMBench split to process (default: en)")
    args = parser.parse_args()
    prepare_mmbench(args.input_dir, args.output_dir, args.split)
