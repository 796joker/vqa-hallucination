"""
Prepare COCO annotations for CHAIR evaluation.
Extracts per-image object category lists from COCO instances annotation file.

Usage:
    python data/prepare_chair_annotations.py \
        --coco_annotation ../downloads/coco/annotations/instances_val2014.json \
        --output_file data/coco_val2014_chair_annots.json

Output format: {"image_id": ["category1", "category2", ...], ...}
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_annotation", type=str, required=True,
                        help="Path to instances_val2014.json")
    parser.add_argument("--output_file", type=str,
                        default="data/coco_val2014_chair_annots.json")
    args = parser.parse_args()

    print(f"Loading COCO annotations from {args.coco_annotation}")
    with open(args.coco_annotation) as f:
        coco = json.load(f)

    # Build category ID -> name mapping
    cat_id_to_name = {}
    for cat in coco["categories"]:
        cat_id_to_name[cat["id"]] = cat["name"]

    print(f"  {len(cat_id_to_name)} categories")

    # Build image_id -> set of category names
    image_objects = {}
    for ann in coco["annotations"]:
        img_id = str(ann["image_id"])
        cat_name = cat_id_to_name[ann["category_id"]]
        if img_id not in image_objects:
            image_objects[img_id] = set()
        image_objects[img_id].add(cat_name)

    # Convert sets to sorted lists
    result = {k: sorted(list(v)) for k, v in image_objects.items()}

    print(f"  {len(result)} images with annotations")
    print(f"  Avg objects per image: {sum(len(v) for v in result.values()) / len(result):.1f}")

    with open(args.output_file, "w") as f:
        json.dump(result, f)
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()
