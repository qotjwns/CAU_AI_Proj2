"""MIHBench questions and scoring, independent of the model (data from scripts/download_mihbench.py)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data" / "mihbench"
TASKS = {  # task -> image folder that its image_list paths are relative to
    "existence_random": "coco_val2014",
    "existence_popular": "coco_val2014",
    "existence_adversarial": "coco_val2014",
    "count": "coco_val2014",
    "id_consistency": "co3d",
}


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
