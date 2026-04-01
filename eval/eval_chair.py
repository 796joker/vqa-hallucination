"""
CHAIR (Caption Hallucination Assessment with Image Relevance) evaluation.
Measures object hallucination in free-form image descriptions.

Metrics:
  - CHAIR_s: fraction of captions containing at least one hallucinated object
  - CHAIR_i: fraction of mentioned objects that are hallucinated
  - Recall: fraction of ground-truth objects mentioned in captions

Reference: Rohrbach et al., "Object Hallucination in Image Captioning" (2018)

Usage:
    # Step 1: Generate captions
    python eval/generate_chair_captions.py \
        --model_path ../downloads/models/Qwen3-VL-8B-Instruct \
        --adapter_path results/sft/lora_r8 \
        --output_file results/eval/sft/chair_captions.json

    # Step 2: Evaluate
    python eval/eval_chair.py \
        --caption_file results/eval/sft/chair_captions.json \
        --annotation_file data/coco_val2014_annots.json \
        --output_file results/eval/sft/chair_metrics.json
"""

import argparse
import json
import re
import string

import nltk

# Ensure NLTK data is available
for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource
                       else f"taggers/{resource}" if "tagger" in resource
                       else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.stem import WordNetLemmatizer

# fmt: off
# MSCOCO 80 categories with synonyms
# Source: lmms-eval/coco_cap_chair, extended with common variants
COCO_SYNONYMS = {
    "person": ["person", "people", "man", "woman", "boy", "girl", "child",
               "kid", "baby", "lady", "gentleman", "guy", "pedestrian",
               "human", "player", "skier", "snowboarder", "surfer",
               "rider", "cyclist", "biker"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "car": ["car", "automobile", "vehicle", "sedan", "suv", "truck", "van"],
    "motorcycle": ["motorcycle", "motorbike"],
    "airplane": ["airplane", "plane", "aircraft", "jet"],
    "bus": ["bus"],
    "train": ["train", "locomotive"],
    "truck": ["truck"],
    "boat": ["boat", "ship", "vessel", "canoe", "kayak", "sailboat", "yacht"],
    "traffic light": ["traffic light", "stoplight", "traffic signal"],
    "fire hydrant": ["fire hydrant", "hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter"],
    "bench": ["bench"],
    "bird": ["bird", "parrot", "pigeon", "crow", "seagull", "eagle",
             "duck", "goose", "owl", "penguin"],
    "cat": ["cat", "kitten", "feline", "tabby"],
    "dog": ["dog", "puppy", "canine", "hound", "poodle", "terrier",
            "retriever", "labrador", "golden retriever", "german shepherd"],
    "horse": ["horse", "pony", "stallion", "mare", "foal"],
    "sheep": ["sheep", "lamb"],
    "cow": ["cow", "cattle", "bull", "calf", "ox"],
    "elephant": ["elephant"],
    "bear": ["bear", "teddy bear", "polar bear", "grizzly"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "backpack": ["backpack", "knapsack", "rucksack", "bag"],
    "umbrella": ["umbrella", "parasol"],
    "handbag": ["handbag", "purse", "clutch"],
    "tie": ["tie", "necktie", "bowtie"],
    "suitcase": ["suitcase", "luggage", "briefcase"],
    "frisbee": ["frisbee"],
    "skis": ["skis", "ski"],
    "snowboard": ["snowboard"],
    "sports ball": ["ball", "sports ball", "baseball", "basketball",
                    "soccer ball", "football", "tennis ball", "volleyball"],
    "kite": ["kite"],
    "baseball bat": ["baseball bat", "bat"],
    "baseball glove": ["baseball glove", "glove", "mitt"],
    "skateboard": ["skateboard"],
    "surfboard": ["surfboard"],
    "tennis racket": ["tennis racket", "racket", "racquet"],
    "bottle": ["bottle"],
    "wine glass": ["wine glass", "wineglass", "glass", "goblet"],
    "cup": ["cup", "mug"],
    "fork": ["fork"],
    "knife": ["knife"],
    "spoon": ["spoon"],
    "bowl": ["bowl"],
    "banana": ["banana"],
    "apple": ["apple"],
    "sandwich": ["sandwich", "sub", "burger", "hamburger"],
    "orange": ["orange"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot"],
    "hot dog": ["hot dog", "hotdog"],
    "pizza": ["pizza"],
    "donut": ["donut", "doughnut"],
    "cake": ["cake", "cupcake", "pastry"],
    "chair": ["chair", "seat", "stool"],
    "couch": ["couch", "sofa", "loveseat"],
    "potted plant": ["potted plant", "plant", "houseplant", "flower pot"],
    "bed": ["bed"],
    "dining table": ["dining table", "table", "desk", "counter"],
    "toilet": ["toilet", "restroom"],
    "tv": ["tv", "television", "monitor", "screen"],
    "laptop": ["laptop", "notebook", "computer"],
    "mouse": ["mouse"],
    "remote": ["remote", "remote control"],
    "keyboard": ["keyboard"],
    "cell phone": ["cell phone", "phone", "cellphone", "smartphone",
                   "mobile phone", "mobile"],
    "microwave": ["microwave"],
    "oven": ["oven", "stove"],
    "toaster": ["toaster"],
    "sink": ["sink", "basin"],
    "refrigerator": ["refrigerator", "fridge"],
    "book": ["book"],
    "clock": ["clock"],
    "vase": ["vase"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "stuffed animal", "plush"],
    "hair drier": ["hair drier", "hair dryer", "blow dryer"],
    "toothbrush": ["toothbrush"],
}
# fmt: on

# Build reverse mapping: synonym -> canonical category
SYNONYM_TO_CATEGORY = {}
for cat, syns in COCO_SYNONYMS.items():
    for syn in syns:
        SYNONYM_TO_CATEGORY[syn] = cat

# Multi-word objects that need special handling
MULTI_WORD_OBJECTS = [
    "traffic light", "fire hydrant", "stop sign", "parking meter",
    "baseball bat", "baseball glove", "tennis racket", "wine glass",
    "hot dog", "potted plant", "dining table", "cell phone",
    "teddy bear", "hair drier", "remote control", "golden retriever",
    "german shepherd", "polar bear", "soccer ball", "tennis ball",
    "sports ball", "flower pot", "mobile phone",
]


def caption_to_objects(caption: str) -> list[str]:
    """Extract COCO object mentions from a caption string."""
    # Strip all <think>/<\/think> tags (DPO models may output malformed tags)
    caption = re.sub(r"</?think>", "", caption)
    caption = caption.strip().lower()

    # Replace multi-word objects with underscored versions for tokenization
    for mw in MULTI_WORD_OBJECTS:
        caption = caption.replace(mw, mw.replace(" ", "_"))

    lemmatizer = WordNetLemmatizer()
    words = nltk.word_tokenize(caption)
    tagged = nltk.pos_tag(words)

    mentioned = set()
    for word, tag in tagged:
        # Restore multi-word objects
        word = word.replace("_", " ")
        # Only consider nouns
        if not tag.startswith("NN"):
            continue
        # Lemmatize
        lemma = lemmatizer.lemmatize(word, "n")
        # Check both original and lemma
        for candidate in [word, lemma]:
            if candidate in SYNONYM_TO_CATEGORY:
                mentioned.add(SYNONYM_TO_CATEGORY[candidate])

    return list(mentioned)


def compute_chair(captions: list[dict], annotations: dict) -> dict:
    """
    Compute CHAIR metrics.

    Args:
        captions: list of {"image_id": int, "caption": str}
        annotations: dict mapping image_id (str or int) -> list of GT object categories
    """
    total_captions = 0
    hallucinated_captions = 0
    total_objects_mentioned = 0
    hallucinated_objects = 0
    total_gt_objects = 0
    recalled_objects = 0

    per_image = []

    for item in captions:
        img_id = str(item["image_id"])
        caption = item["caption"]
        gt_objects = set(annotations.get(img_id, []))

        if not gt_objects:
            continue

        mentioned = caption_to_objects(caption)
        mentioned_set = set(mentioned)

        total_captions += 1
        total_objects_mentioned += len(mentioned_set)
        total_gt_objects += len(gt_objects)

        # Hallucinated objects
        halluc = mentioned_set - gt_objects
        hallucinated_objects += len(halluc)

        if halluc:
            hallucinated_captions += 1

        # Recall
        recalled = mentioned_set & gt_objects
        recalled_objects += len(recalled)

        per_image.append({
            "image_id": img_id,
            "mentioned": list(mentioned_set),
            "gt_objects": list(gt_objects),
            "hallucinated": list(halluc),
            "recalled": list(recalled),
        })

    chair_s = hallucinated_captions / total_captions if total_captions else 0
    chair_i = hallucinated_objects / total_objects_mentioned if total_objects_mentioned else 0
    recall = recalled_objects / total_gt_objects if total_gt_objects else 0

    return {
        "CHAIR_s": round(chair_s * 100, 2),
        "CHAIR_i": round(chair_i * 100, 2),
        "Recall": round(recall * 100, 2),
        "total_captions": total_captions,
        "total_objects_mentioned": total_objects_mentioned,
        "hallucinated_objects": hallucinated_objects,
        "hallucinated_captions": hallucinated_captions,
        "per_image": per_image,
    }


def print_chair(metrics: dict, label: str = ""):
    if label:
        print(f"\n{'=' * 50}")
        print(f"  {label}")
        print(f"{'=' * 50}")
    print(f"  CHAIR_s: {metrics['CHAIR_s']:.2f}%  (captions with hallucination)")
    print(f"  CHAIR_i: {metrics['CHAIR_i']:.2f}%  (hallucinated object ratio)")
    print(f"  Recall:  {metrics['Recall']:.2f}%  (GT objects mentioned)")
    print(f"  Total captions: {metrics['total_captions']}")
    print(f"  Total objects mentioned: {metrics['total_objects_mentioned']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption_file", type=str, required=True,
                        help="JSON file with captions: [{image_id, caption}, ...]")
    parser.add_argument("--annotation_file", type=str, required=True,
                        help="JSON file mapping image_id -> [object categories]")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    with open(args.caption_file) as f:
        captions = json.load(f)

    with open(args.annotation_file) as f:
        annotations = json.load(f)

    metrics = compute_chair(captions, annotations)
    print_chair(metrics, label=args.caption_file)

    if args.output_file:
        # Save without per_image detail (too large)
        save_metrics = {k: v for k, v in metrics.items() if k != "per_image"}
        with open(args.output_file, "w") as f:
            json.dump(save_metrics, f, indent=2)
        print(f"\nMetrics saved to {args.output_file}")
