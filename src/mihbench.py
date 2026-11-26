"""MIHBench questions and scoring, independent of the model (data from scripts/download_mihbench.py)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "mihbench"
TASKS = {  # task -> image folder that its image_list paths are relative to
    "existence_random": "coco_val2014",
    "existence_popular": "coco_val2014",
    "existence_adversarial": "coco_val2014",
    "count": "coco_val2014",
    "id_consistency": "co3d",
}
# The 80 COCO category names (instances_*2014.json, in id order); existence questions use them as they are.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
EXISTENCE_QUESTION = re.compile(r"Is there an? (.+?) in (?:the imange\? in )?all \d+ images\?")  # 86 have the typo
COUNT_QUESTION = re.compile(r"Are there the same number of (.+?) in image 1 and image 2\?")


def count_plural(name):
    """A class as count questions write it: 'people' for person, else + 'es' after s, sh, ch or x, else + 's',
    applied mechanically ('knifes', 'mouses', 'sheeps', 'scissorses')."""
    if name == "person":
        return "people"
    return name + ("es" if name.endswith(("s", "sh", "ch", "x")) else "s")


COUNT_NOUNS = {count_plural(c): c for c in COCO_CLASSES}


def question_class(task, text):
    """The COCO class an existence or count question asks about; None for ID consistency, whose questions are
    about CO3D objects ('Is there a same kite in all 4 images?'). ValueError if no COCO class is found."""
    if task == "id_consistency":
        return None
    m = (COUNT_QUESTION if task == "count" else EXISTENCE_QUESTION).fullmatch(text)
    name = m and (COUNT_NOUNS.get(m.group(1)) if task == "count" else m.group(1))
    if name not in COCO_CLASSES:
        raise ValueError(f"{task}: no COCO class in {text!r}")
    return name


def load_questions(task):
    """Rows of questions/<task>.jsonl, each with an added 'image_paths' list."""
    image_dir = ROOT / "images" / TASKS[task]
    with open(ROOT / "questions" / f"{task}.jsonl") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        row["image_paths"] = [image_dir / name for name in row["image_list"]]
    return rows


def parse_yes_no(text):
    """The official rule (eval.py in pgtrece/DAB): 'no' if the first sentence has the word No, no or not."""
    words = text.split(".")[0].replace(",", "").split(" ")
    return "no" if {"No", "no", "not"} & set(words) else "yes"


def score(rows):
    """Metrics of the official eval.py, with 'yes' as the positive class. rows: dicts with 'pred' and 'label'."""
    pairs = [(r["pred"] == "yes", r["label"].lower() == "yes") for r in rows]
    tp = sum(p and t for p, t in pairs)
    fp = sum(p and not t for p, t in pairs)
    tn = sum(not p and not t for p, t in pairs)
    fn = sum(not p and t for p, t in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(pairs),
        "accuracy": (tp + tn) / len(pairs) if pairs else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "yes_ratio": (tp + fp) / len(pairs) if pairs else 0.0,
    }
