#!/usr/bin/env python3
"""
    Step 1 of the COCO existence data: how many train2014 images each class can use.
    Per (image, class), from instances_train2014.json:
      P         a non-crowd instance of the class covers >= 1% of the image (segmentation area)
      excluded  the class is there but under 1% or crowd-only: used as neither P nor N for that class
      N         no instance of the class (images without any annotation are never used)
      hard N    an N image holding one of the class's companions: the 3 classes with the highest lift among those
                seen with it in >= 30 images and with lift > 1, never a class found in > 25% of images (person,
                55%: "has a person" says nothing). A class with none (bear) uses the other classes of its
                supercategory instead.
    Need per class (30/6/12 triplets, half positive-anchor (4 P + 1 N), half negative-anchor (3 P + 2 N)):
      P slots train 105, dev 21, test 42 -> 168 images without reuse, 98 if train images are used up to 3 times
    hair drier and toaster fall short and are left out of training (EXCLUDED_CLASSES): 78 classes remain.
    Counts are per class, before images are split globally, so they are upper bounds.
      data/coco_existence/pool_stats.json   counts, companions, shortfalls, MIHBench leakage and object sizes
    python src/coco_pools.py
"""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
COCO = PROJECT / "data" / "coco"
MIHBENCH = PROJECT / "data" / "mihbench" / "questions"
OUT = PROJECT / "data" / "coco_existence" / "pool_stats.json"

MIN_AREA = 0.01
MIN_COOC = 30
MAX_COMPANION_SHARE = 0.25
HARD_TOP_K = 3
EXCLUDED_CLASSES = ("hair drier", "toaster")  # P short even with train reuse (pool_stats 2026-11-26)
TRIPLETS = {"train": 30, "dev": 6, "test": 12}
TRAIN_REUSE = 3
PER_TRIPLET = {"P": 3.5, "N": 1.5}  # mean of positive-anchor (4 P, 1 N) and negative-anchor (3 P, 2 N)
MIH_COCO = re.compile(r"COCO_val2014_(\d{12})\.jpg")
MIH_QUESTION = re.compile(r"Is there an? (.+?) in (?:the imange\? in )?all \d+ images\?")  # 86 have the typo


def load_coco(path):
    """images {id: info}, classes {id: category}, and per annotated image {class: largest non-crowd area ratio},
    where 0.0 means the class is there only as crowd regions."""
    with open(path) as f:
        d = json.load(f)
    images = {im["id"]: im for im in d["images"]}
    classes = {c["id"]: c for c in d["categories"]}
    present = defaultdict(dict)
    for a in d["annotations"]:
        im = images[a["image_id"]]
        ratio = 0.0 if a["iscrowd"] else a["area"] / (im["width"] * im["height"])
        cls = present[a["image_id"]]
        cls[a["category_id"]] = max(cls.get(a["category_id"], 0.0), ratio)
    return images, classes, dict(present)


def companions(classes, present):
    """{class: ([(companion, images with both, lift)], "lift" or "supercategory")}, the classes whose images
    serve as hard N for it."""
    n_images = len(present)
    count = Counter(c for cls in present.values() for c in cls)
    co = Counter((a, b) for cls in present.values() for a in cls for b in cls if a != b)
    pair = lambda c, b: (b, co[c, b], co[c, b] * n_images / (count[c] * count[b]))
    allowed = [b for b in classes if count[b] <= MAX_COMPANION_SHARE * n_images]
    out = {}
    for c in classes:
        cand = [pair(c, b) for b in allowed if b != c and co[c, b] >= MIN_COOC]
        cand = sorted((x for x in cand if x[2] > 1), key=lambda x: -x[2])[:HARD_TOP_K]
        if cand:
            out[c] = (cand, "lift")
        else:
            sup = classes[c]["supercategory"]
            out[c] = ([pair(c, b) for b in allowed if b != c and classes[b]["supercategory"] == sup], "supercategory")
    return out


def build_pools(classes, present, comp):
    """{class: {"P", "excluded", "N", "hard_N": [image ids]}}."""
    pools = {c: {"P": [], "excluded": [], "N": []} for c in classes}
    for i, cls in present.items():
        for c in classes:
            r = cls.get(c)
            pools[c]["N" if r is None else "P" if r >= MIN_AREA else "excluded"].append(i)
    for c in classes:
        with_ = {b for b, _, _ in comp[c][0]}
        pools[c]["hard_N"] = [i for i in pools[c]["N"] if with_ & present[i].keys()]
    return pools


def need(kind):
    """Slots per split, the minimum number of distinct images (train reuse allowed) and the no-reuse number."""
    slots = {s: math.ceil(n * PER_TRIPLET[kind]) for s, n in TRIPLETS.items()}
    return {**slots, "min": slots["dev"] + slots["test"] + math.ceil(slots["train"] / TRAIN_REUSE),
            "no_reuse": sum(slots.values())}


def size_summary(ratios):
    r = sorted(ratios)
    at = lambda p: round(r[min(len(r) - 1, int(p * len(r)))], 4)
    return {"pairs": len(r), "under_1pct": round(sum(x < MIN_AREA for x in r) / len(r), 3),
            "crowd_only": sum(x == 0.0 for x in r),
            "quantiles": {f"p{int(p * 100)}": at(p) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}}


def mihbench_rows(prefix=""):
    for path in sorted(MIHBENCH.glob(f"{prefix}*.jsonl")):
        with open(path) as f:
            yield from (json.loads(line) for line in f if line.strip())


def mihbench_object_sizes(classes):
    """Area ratio of the asked object in MIHBench existence images that contain it (unique image-class pairs)."""
    val_path = COCO / "annotations" / "instances_val2014.json"
    if not val_path.exists():
        return {"skipped": f"{val_path.relative_to(PROJECT)} not found: python scripts/download_coco_train2014.py"}
    _, _, present = load_coco(val_path)
    by_name = {c["name"]: i for i, c in classes.items()}
    pairs = {}
    for q in mihbench_rows("existence_"):
        c = by_name[MIH_QUESTION.fullmatch(q["text"]).group(1)]
        for name in q["image_list"]:
            i = int(MIH_COCO.fullmatch(name).group(1))
            if c in present.get(i, {}):
                pairs[(i, c)] = present[i][c]
    return size_summary(pairs.values())


def mihbench_existence_flags():
    """{(task, question_id): (label_consistent, large_objects)} for the MIHBench existence questions, judged with
    the val2014 annotations, or None if instances_val2014.json is missing.
      label_consistent  the label is yes exactly when every image holds the asked class
      large_objects     each image lacks the class or holds it at >= 1% (the training P/N definition)"""
    val_path = COCO / "annotations" / "instances_val2014.json"
    if not val_path.exists():
        return None
    _, classes, present = load_coco(val_path)
    by_name = {c["name"]: i for i, c in classes.items()}
    flags = {}
    for path in sorted(MIHBENCH.glob("existence_*.jsonl")):
        with open(path) as f:
            for q in (json.loads(line) for line in f if line.strip()):
                c = by_name[MIH_QUESTION.fullmatch(q["text"]).group(1)]
                r = [present.get(int(MIH_COCO.fullmatch(n).group(1)), {}).get(c) for n in q["image_list"]]
                flags[path.stem, q["question_id"]] = ((q["label"] == "yes") == all(x is not None for x in r),
                                                      all(x is None or x >= MIN_AREA for x in r))
    return flags


def main():
    images, classes, present = load_coco(COCO / "annotations" / "instances_train2014.json")
    comp = companions(classes, present)
    pools = build_pools(classes, present, comp)
    need_p, need_n = need("P"), need("N")
    need_hard = math.ceil(need_n["min"] / 2)

    rows = []
    for c, cat in classes.items():
        p = pools[c]
        n_p, n_n, n_h = len(p["P"]), len(p["N"]), len(p["hard_N"])
        status = "ok" if n_p >= need_p["no_reuse"] else "reuse" if n_p >= need_p["min"] else "short"
        if n_n < need_n["min"] or n_h < need_hard:
            status = "short"
        rows.append({"id": c, "name": cat["name"], "supercategory": cat["supercategory"],
                     "used": cat["name"] not in EXCLUDED_CLASSES, "status": status,
                     "P": n_p, "excluded": len(p["excluded"]), "N": n_n, "hard_N": n_h,
                     "P_short_by": max(0, need_p["min"] - n_p), "hard_source": comp[c][1],
                     "hard_with": [{"class": classes[b]["name"], "images_with_both": n, "lift": round(lift, 2)}
                                   for b, n, lift in comp[c][0]]})
    rows.sort(key=lambda x: x["P"])
    short = sorted(x["name"] for x in rows if x["status"] == "short")
    if short != sorted(EXCLUDED_CLASSES):
        print(f"warning: short classes {short} differ from EXCLUDED_CLASSES {sorted(EXCLUDED_CLASSES)}")

    mih_ids = {int(m.group(1)) for q in mihbench_rows() for n in q["image_list"] if (m := MIH_COCO.fullmatch(n))}
    missing = [i for i in present if not (COCO / "train2014" / images[i]["file_name"]).is_file()]
    used = [x for x in rows if x["used"]]
    stats = {
        "params": {"min_area": MIN_AREA, "min_cooccurrence_images": MIN_COOC,
                   "max_companion_share": MAX_COMPANION_SHARE, "hard_top_k": HARD_TOP_K,
                   "triplets_per_class": TRIPLETS, "train_reuse": TRAIN_REUSE},
        "need": {"P": need_p, "N": need_n, "hard_N_min": need_hard},
        "classes_used": len(used),
        "excluded_classes": [{"name": x["name"], "P": x["P"], "P_short_by": x["P_short_by"]}
                             for x in rows if not x["used"]],
        "images": {"train2014": len(images), "annotated": len(present), "no_annotation": len(images) - len(present),
                   "missing_files": len(missing)},
        "leakage": {"mihbench_coco_images": len(mih_ids), "overlap_with_train2014": len(mih_ids & images.keys())},
        "object_size": {"train2014_all_pairs": size_summary(r for cls in present.values() for r in cls.values()),
                        "mihbench_existence": mihbench_object_sizes(classes)},
        "classes": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, indent=2) + "\n")

    print(f"{'class':<16}{'P':>7}{'excl':>7}{'N':>8}{'hardN':>7}  {'status':<8} hard N companions")
    for x in rows:
        status = x["status"] if x["used"] else "dropped"
        with_ = ", ".join(f"{h['class']} x{h['lift']}" for h in x["hard_with"])
        if x["hard_source"] == "supercategory":
            with_ = f"[{x['supercategory']} fallback] " + ", ".join(h["class"] for h in x["hard_with"])
        print(f"{x['name']:<16}{x['P']:>7}{x['excluded']:>7}{x['N']:>8}{x['hard_N']:>7}  {status:<8} {with_}")
    print(f"\nclasses used: {len(used)}; dropped: {stats['excluded_classes']}")
    print(f"need per class: P {need_p['no_reuse']} without reuse, {need_p['min']} with train reuse <= {TRAIN_REUSE}; "
          f"hard N >= {need_hard}")
    print(f"images: {stats['images']}")
    print(f"leakage: {stats['leakage']}")
    print(f"object size, train2014: {stats['object_size']['train2014_all_pairs']}")
    print(f"object size, MIHBench existence: {stats['object_size']['mihbench_existence']}")
    print(f"\n{OUT}")


if __name__ == "__main__":
    main()
