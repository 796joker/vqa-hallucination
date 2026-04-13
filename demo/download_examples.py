"""
Download example images for the Gradio demo.

Downloads a curated set of COCO val2014 images used in case studies.
Run this before launching demo/app.py on a cloud platform.

Usage:
    python demo/download_examples.py
    python demo/download_examples.py --mirror  # Use mirror if COCO CDN is slow
"""

import os
import sys
import argparse
import urllib.request
import urllib.error

COCO_BASE_URL = "http://images.cocodataset.org/val2014"
MIRROR_URL = "https://hf-mirror.com/datasets/merve/coco/resolve/main/val2014"

# 10 images matching results/case_studies/case_studies.json
EXAMPLE_IMAGES = [
    {"id": "000000012896", "desc": "Indoor living room scene"},
    {"id": "000000125245", "desc": "Street scene with vehicles"},
    {"id": "000000139475", "desc": "People in outdoor setting"},
    {"id": "000000223174", "desc": "Kitchen / dining scene"},
    {"id": "000000289173", "desc": "Animals in scene"},
    {"id": "000000298067", "desc": "Urban street view"},
    {"id": "000000331844", "desc": "Indoor scene with furniture"},
    {"id": "000000423058", "desc": "Sports / outdoor activity"},
    {"id": "000000480140", "desc": "Complex multi-object scene"},
    {"id": "000000483742", "desc": "Food / tabletop scene"},
]


def main():
    parser = argparse.ArgumentParser(description="Download COCO example images for demo")
    parser.add_argument("--mirror", action="store_true",
                        help="Use HuggingFace mirror instead of COCO CDN")
    parser.add_argument("--output", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples"),
                        help="Output directory (default: demo/examples/)")
    args = parser.parse_args()

    base_url = MIRROR_URL if args.mirror else COCO_BASE_URL
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # Clean broken symlinks
    cleaned = 0
    for f in os.listdir(out_dir):
        path = os.path.join(out_dir, f)
        if os.path.islink(path) and not os.path.exists(path):
            os.unlink(path)
            cleaned += 1
    if cleaned:
        print(f"Removed {cleaned} broken symlinks")

    downloaded, skipped, failed = 0, 0, 0

    for img in EXAMPLE_IMAGES:
        filename = f"COCO_val2014_{img['id']}.jpg"
        filepath = os.path.join(out_dir, filename)

        if os.path.isfile(filepath) and os.path.getsize(filepath) > 1000:
            print(f"  [skip] {filename} (already exists)")
            skipped += 1
            continue

        url = f"{base_url}/{filename}"
        print(f"  [download] {filename} ... ", end="", flush=True)
        try:
            urllib.request.urlretrieve(url, filepath)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"OK ({size_kb:.0f} KB)")
            downloaded += 1
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f"FAILED: {e}")
            if os.path.exists(filepath):
                os.unlink(filepath)
            failed += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    if failed:
        print("Tip: try --mirror flag if COCO CDN is not accessible")
        sys.exit(1)


if __name__ == "__main__":
    main()
