#!/usr/bin/env python3
"""
    Step 2 of the COCO existence data: build the 4-case counterfactual groups, check them, write them.
      data/coco_existence/train.jsonl, dev.jsonl, test.jsonl   one question per line, 4 per group
      data/coco_existence/pilot_train.jsonl                    6 train groups per class (one per position pair)
      data/coco_existence/manifest.json                        settings, seed, input / output sha256, leakage checks
      data/coco_existence/statistics.json                      counts, balance, image reuse, check results
      data/coco_existence/excluded_images.json                 images left out, and why
    A group is 4 P images and 1 N image of one class in 4 questions. P1 stays at the fixed position r; the stay
    position s holds P2 (A, B) or P4 (C, D); the flip position f holds P3 (A, C) or N1 (B, D):
      A = P1 P2 P3 -> Yes    B = P1 P2 N1 -> No    C = P1 P4 P3 -> Yes    D = P1 P4 N1 -> No   (shown for r, s, f)
    Flip pairs (answer changes): A-B, C-D. Stay pairs (answer kept): A-C, B-D. The 6 (f, s) position pairs are
    spread evenly per class and split. 78 classes; 30 / 6 / 12 groups per class for train / dev / test.
    P, N and hard N as in src/coco_pools.py; half of the N images are hard N. No image is in two splits;
    a train image is in at most 3 groups, a dev or test image in one. "question" is the bare question with
    MIHBench's article rule (an before a vowel); src/cf_data.py adds the answer instruction for training.
    Images listed in data/coco_existence/exclude/*.txt (one COCO id or file name per line) are left out.
    python src/build_coco_existence.py           # build, check, write
    python src/build_coco_existence.py --check   # rebuild in memory and compare with the written files
"""
import argparse
import datetime
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from coco_pools import (COCO, EXCLUDED_CLASSES, HARD_TOP_K, MAX_COMPANION_SHARE, MIH_COCO, MIN_AREA, MIN_COOC,
                        TRIPLETS, build_pools, companions, load_coco, mihbench_rows)

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
OUT = DATA / "coco_existence"
EXCLUDE_DIR = OUT / "exclude"
ANN = COCO / "annotations" / "instances_train2014.json"

SEED = 0
GROUPS = TRIPLETS  # groups per class and split: train 30, dev 6, test 12
CAP = {"train": 3, "dev": 1, "test": 1}  # groups an image may be in
REUSE_PREFERENCE = 0.5  # train: chance of first trying an image already in train (fewer distinct images)
SOURCE = "COCO train2014"
SPLITS = ("train", "dev", "test")
BUILD_ORDER = ("test", "dev", "train")  # per class, so the scarce images go to the no-reuse splits first
CASES = ("A", "B", "C", "D")
FLIP_PARTNER = {"A": "B", "B": "A", "C": "D", "D": "C"}
STAY_PARTNER = {"A": "C", "C": "A", "B": "D", "D": "B"}
POSITION_PAIRS = [(f, s) for f in range(3) for s in range(3) if f != s]  # (flip position, stay position)
POOL = {"P": "P", "random": "N", "hard": "hard_N"}


def question(name):
    """MIHBench's wording: 'an' before a vowel (umbrella, apple, elephant), COCO names as they are ('a skis')."""
    return f"Is there {'an' if name[0] in 'aeiou' else 'a'} {name} in all 3 images?"


class Shortage(Exception):
    pass


class Builder:
    def __init__(self, seed, classes, present, pools, comp):
        self.rng = random.Random(seed)
        self.classes, self.present, self.pools = classes, present, pools
        self.companions = {c: {b for b, _, _ in comp[c][0]} for c in classes}
        self.split_of, self.uses, self.pos = {}, Counter(), defaultdict(list)
        self.train_open = set()  # train images used in fewer than CAP["train"] groups
        self.groups, self.seen = [], set()

    def fits(self, kind, i, c):
        cls = self.present[i]
        if kind == "P":
            return cls.get(c, -1.0) >= MIN_AREA
        return c not in cls and (kind == "random" or not self.companions[c].isdisjoint(cls))

    def fresh(self, kind, c, taken):
        pool = self.pools[c][POOL[kind]]
        if not pool:
            return None
        for _ in range(50):
            i = self.rng.choice(pool)
            if i not in self.split_of and i not in taken:
                return i
        rest = [i for i in pool if i not in self.split_of and i not in taken]
        return self.rng.choice(rest) if rest else None

    def reused(self, kind, c, taken, k):
        cand = [i for i in sorted(self.train_open) if i not in taken and self.fits(kind, i, c)]
        cand = [i for i in cand if k not in self.pos[i]] or cand  # prefer a position the image has not had
        return self.rng.choice(cand) if cand else None

    def pick(self, kind, c, split, taken, k):
        i = None
        if split == "train" and self.rng.random() < REUSE_PREFERENCE:
            i = self.reused(kind, c, taken, k)
        if i is None:
            i = self.fresh(kind, c, taken)
        if i is None and split == "train":
            i = self.reused(kind, c, taken, k)
        if i is None:
            raise Shortage(f"{self.classes[c]['name']}: no {kind} image left for {split}")
        taken.add(i)
        return i

    def use(self, i, split, p):
        self.split_of[i] = split
        self.uses[i] += 1
        self.pos[i].append(p)
        if split == "train":
            (self.train_open.add if self.uses[i] < CAP["train"] else self.train_open.discard)(i)

    def group(self, c, split, f, s, n_kind):
        """{case: [3 image ids]} for one group with flip position f and stay position s."""
        r = 3 - f - s
        for _ in range(20):
            taken = set()
            img = {"P1": self.pick("P", c, split, taken, r), "P2": self.pick("P", c, split, taken, s),
                   "P4": self.pick("P", c, split, taken, s), "P3": self.pick("P", c, split, taken, f),
                   "N1": self.pick(n_kind, c, split, taken, f)}
            cases = {}
            for case, (at_s, at_f) in {"A": ("P2", "P3"), "B": ("P2", "N1"), "C": ("P4", "P3"), "D": ("P4", "N1")}.items():
                row = [0, 0, 0]
                row[r], row[s], row[f] = img["P1"], img[at_s], img[at_f]
                cases[case] = row
            keys = {(c, tuple(v)) for v in cases.values()}
            if not keys & self.seen:
                break
        else:
            raise Shortage(f"{self.classes[c]['name']}: could not avoid a duplicate question in {split}")
        self.seen |= keys
        for name, p in (("P1", r), ("P2", s), ("P4", s), ("P3", f), ("N1", f)):
            self.use(img[name], split, p)
        return cases

    def build_class(self, c):
        name = self.classes[c]["name"]
        for split in BUILD_ORDER:
            n = GROUPS[split]
            plan = [fs for fs in POSITION_PAIRS for _ in range(n // len(POSITION_PAIRS))]
            self.rng.shuffle(plan)
            kinds = ["hard"] * (n // 2) + ["random"] * (n - n // 2)  # one N image per group
            self.rng.shuffle(kinds)
            for t, ((f, s), kind) in enumerate(zip(plan, kinds)):
                self.groups.append({"split": split, "class_id": c, "class_name": name, "f": f, "s": s,
                                    "n_kind": kind, "group_id": f"{split}-{slug(name)}-{t:02d}",
                                    "cases": self.group(c, split, f, s, kind)})


def slug(name):
    return name.replace(" ", "_")


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def global_exclusions(images, present):
    """{reason: sorted image ids} for images used in no split."""
    out = {"no_annotation": sorted(set(images) - present.keys())}
    mih = {int(m.group(1)) for q in mihbench_rows() for n in q["image_list"] if (m := MIH_COCO.fullmatch(n))}
    out["mihbench"] = sorted(mih & images.keys())
    out["missing_file"] = sorted(i for i in present if not (COCO / "train2014" / images[i]["file_name"]).is_file())
    by_name = {im["file_name"]: i for i, im in images.items()}
    for path in sorted(EXCLUDE_DIR.glob("*.txt")):
        ids, unknown = set(), []
        for tok in path.read_text().split():
            i = int(tok) if tok.isdigit() else by_name.get(tok)
            (ids.add(i) if i in images else unknown.append(tok))
        if unknown:
            sys.exit(f"{path}: not COCO train2014 images: {unknown[:5]}")
        out[f"exclude/{path.name}"] = sorted(ids)
    return out


def to_rows(groups, images, present):
    rows = {s: [] for s in SPLITS}
    for g in sorted(groups, key=lambda g: (SPLITS.index(g["split"]), g["class_id"], g["group_id"])):
        c = g["class_id"]
        for case in CASES:
            ids = g["cases"][case]
            labels = [1 if present[i].get(c, -1.0) >= MIN_AREA else 0 for i in ids]
            rows[g["split"]].append({
                "case_id": f"{g['group_id']}-{case}", "group_id": g["group_id"], "split": g["split"], "case": case,
                "class_id": c, "class_name": g["class_name"], "question": question(g["class_name"]),
                "images": [f"coco/train2014/{images[i]['file_name']}" for i in ids], "image_ids": ids,
                "image_labels": labels, "gt": "Yes" if all(labels) else "No",
                "flip_position": g["f"], "stay_position": g["s"], "fixed_position": 3 - g["f"] - g["s"],
                "flip_partner": f"{g['group_id']}-{FLIP_PARTNER[case]}",
                "stay_partner": f"{g['group_id']}-{STAY_PARTNER[case]}",
                "negative_positions": [p for p, x in enumerate(labels) if x == 0],
                "negative_types": [g["n_kind"]] if case in ("B", "D") else [], "source": SOURCE})
    return rows


def pick_pilot(rows):
    """One train group per class x (flip, stay) position pair (6 per class)."""
    rng = random.Random(SEED + 1)
    by = defaultdict(list)
    for r in rows["train"]:
        if r["case"] == "A":
            by[r["class_id"], r["flip_position"], r["stay_position"]].append(r["group_id"])
    chosen = {rng.choice(sorted(v)) for _, v in sorted(by.items())}
    return [r for r in rows["train"] if r["group_id"] in chosen]


def run_checks(rows, pilot, present, comp_sets, excluded, used_classes):
    """{check: "pass" or the first problems found}."""
    res = {}

    def check(name, problems):
        res[name] = "pass" if not problems else [str(p) for p in problems[:5]]

    by_g = defaultdict(dict)
    for s in SPLITS:
        for r in rows[s]:
            by_g[r["group_id"]][r["case"]] = r
    check("four cases per group", [g for g, d in by_g.items() if sorted(d) != list(CASES)])
    bad = []
    for g, d in by_g.items():
        a = d["A"]
        f, s, r = a["flip_position"], a["stay_position"], a["fixed_position"]
        same = all(x["class_id"] == a["class_id"] and (x["flip_position"], x["stay_position"]) == (f, s)
                   for x in d.values()) and f != s and r == 3 - f - s
        only = lambda x, y, p: x["image_ids"][p] != y["image_ids"][p] and all(
            x["image_ids"][q] == y["image_ids"][q] for q in range(3) if q != p)
        pairs = only(d["A"], d["B"], f) and only(d["C"], d["D"], f) and only(d["A"], d["C"], s) and only(d["B"], d["D"], s)
        five = len({i for x in d.values() for i in x["image_ids"]}) == 5
        if not (same and pairs and five):
            bad.append(g)
    check("flip pairs differ only at the flip position, stay pairs only at the stay position, 5 images", bad)
    bad = []
    for s in SPLITS:
        for r in rows[s]:
            c = r["class_id"]
            for i, lab in zip(r["image_ids"], r["image_labels"]):
                area = present[i].get(c)
                if not ((lab == 1 and area is not None and area >= MIN_AREA) or (lab == 0 and area is None)):
                    bad.append((r["case_id"], i))
            if r["gt"] != ("Yes" if all(r["image_labels"]) else "No"):
                bad.append(r["case_id"])
    check("labels follow the annotations (P >= 1%, N absent, no 0-1% image)", bad)
    case_ids = {r["case_id"]: r for s in SPLITS for r in rows[s]}
    bad = [g for g, d in by_g.items() if [d[x]["gt"] for x in CASES] != ["Yes", "No", "Yes", "No"]]
    bad += [r["case_id"] for r in case_ids.values()
            if case_ids[r["flip_partner"]]["gt"] == r["gt"] or case_ids[r["stay_partner"]]["gt"] != r["gt"]]
    check("answers A Yes, B No, C Yes, D No; flip partner opposite, stay partner same", bad)
    bad = [(r["case_id"], i) for s in SPLITS for r in rows[s] for p, kind in zip(r["negative_positions"],
           r["negative_types"]) if kind == "hard" and comp_sets[r["class_id"]].isdisjoint(
           present[i := r["image_ids"][p]])]
    check("hard N images hold a companion class", bad)
    check("question text", [r["case_id"] for r in case_ids.values() if r["question"] != question(r["class_name"])])
    split_imgs = {s: {i for r in rows[s] for i in r["image_ids"]} for s in SPLITS}
    check("splits share no image", [(a, b, len(split_imgs[a] & split_imgs[b]))
                                    for a in SPLITS for b in SPLITS if a < b and split_imgs[a] & split_imgs[b]])
    bad = []
    for s in SPLITS:
        per_img = Counter(i for g in {r["group_id"] for r in rows[s]}
                          for i in {j for d in by_g[g].values() for j in d["image_ids"]})
        bad += [(s, i, n) for i, n in per_img.items() if n > CAP[s]]
    check("image reuse caps (train 3, dev 1, test 1 groups)", bad)
    bad = []
    for s in SPLITS:
        a_rows = [r for r in rows[s] if r["case"] == "A"]
        per = Counter((r["class_id"], r["flip_position"], r["stay_position"]) for r in a_rows)
        want = GROUPS[s] // len(POSITION_PAIRS)
        bad += [(s, k, v) for k, v in per.items() if v != want]
        bad += [(s, c) for c in used_classes if sum(v for k, v in per.items() if k[0] == c) != GROUPS[s]]
        yes = Counter((r["class_id"], r["gt"]) for r in rows[s])
        bad += [(s, c, "yes/no") for c in used_classes if yes[c, "Yes"] != yes[c, "No"]]
        npos = Counter((r["class_id"], p) for r in rows[s] for p in r["negative_positions"])
        bad += [(s, c, "N position") for c in used_classes if len({npos[c, p] for p in range(3)}) != 1]
        hard = Counter((r["class_id"], r["negative_types"][0]) for r in rows[s] if r["case"] == "B")
        bad += [(s, c, "hard/random") for c in used_classes if hard[c, "hard"] != hard[c, "random"]]
    check("per class: groups, 6 position pairs even, yes = no, N positions even, hard = random", bad)
    keys = Counter((r["class_id"], tuple(r["image_ids"])) for r in case_ids.values())
    check("no duplicate question", [k for k, v in keys.items() if v > 1])
    drop = set().union(*excluded.values())
    check("excluded images unused", sorted(drop & set().union(*split_imgs.values())))
    check("classes", sorted({r["class_name"] for r in case_ids.values()} & set(EXCLUDED_CLASSES))
          + ([] if len({r["class_id"] for r in rows["train"]}) == len(used_classes) else ["class count"]))
    per = Counter((r["class_id"], r["flip_position"], r["stay_position"]) for r in pilot if r["case"] == "A")
    train_ids = {r["case_id"] for r in rows["train"]}
    check("pilot: 6 train groups per class, one per position pair",
          [k for k, v in per.items() if v != 1] + ([] if len(per) == 6 * len(used_classes) else ["count"])
          + [r["case_id"] for r in pilot if r["case_id"] not in train_ids])
    return res


def unreadable(paths):
    def bad(p):
        try:
            with Image.open(p) as im:
                im.load()
            return None
        except Exception as e:  # truncated or corrupt JPEG
            return f"{p.name}: {e}"
    with ThreadPoolExecutor(16) as ex:
        return [b for b in ex.map(bad, paths) if b]


def statistics(rows, pilot, present):
    def split_stats(rs):
        members = defaultdict(dict)  # group -> image -> position
        for r in rs:
            for p, i in enumerate(r["image_ids"]):
                members[r["group_id"]][i] = p
        per_img, positions = Counter(), defaultdict(set)
        for g in members.values():
            for i, p in g.items():
                per_img[i] += 1
                positions[i].add(p)
        reused = [i for i, n in per_img.items() if n > 1]
        config = Counter("".join("P" if x else "N" for x in r["image_labels"]) for r in rs)
        sizes = sorted(present[i][r["class_id"]] for r in rs for i, x in zip(r["image_ids"], r["image_labels"]) if x)
        q = lambda p: round(sizes[int(p * (len(sizes) - 1))], 4)
        return {"groups": len(members), "questions": len(rs), "unique_images": len(per_img),
                "yes": sum(r["gt"] == "Yes" for r in rs), "no": sum(r["gt"] == "No" for r in rs),
                "configurations": dict(sorted(config.items())),
                "position_pairs_flip_stay": {f"{f}{s}": n for (f, s), n in sorted(Counter(
                    (r["flip_position"], r["stay_position"]) for r in rs if r["case"] == "A").items())},
                "negative_types": dict(Counter(r["negative_types"][0] for r in rs if r["case"] == "B")),
                "groups_per_image": dict(sorted(Counter(per_img.values()).items())),
                "reused_images": len(reused),
                "reused_images_always_same_position": sum(len(positions[i]) == 1 for i in reused),
                "P_object_area_quantiles": {f"p{int(p * 100)}": q(p) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}}
    per_class = defaultdict(dict)
    for s in SPLITS:
        for r in rows[s]:
            d = per_class[r["class_name"]].setdefault(s, {"P": set(), "N": set()})
            for i, x in zip(r["image_ids"], r["image_labels"]):
                d["P" if x else "N"].add(i)
    return {"splits": {s: split_stats(rows[s]) for s in SPLITS}, "pilot_train": split_stats(pilot),
            "unique_images_per_class": {c: {s: {k: len(v) for k, v in d.items()} for s, d in sorted(v.items())}
                                        for c, v in sorted(per_class.items())}}


def build(seed=SEED, extra_drop=()):
    images, classes, present_all = load_coco(ANN)
    comp = companions(classes, present_all)
    excluded = global_exclusions(images, present_all)
    if extra_drop:
        excluded["unreadable"] = sorted(extra_drop)
    drop = set().union(*excluded.values())
    present = {i: cls for i, cls in present_all.items() if i not in drop}
    pools = build_pools(classes, present, comp)
    used = [c for c in classes if classes[c]["name"] not in EXCLUDED_CLASSES]
    order = sorted(used, key=lambda c: (min(len(pools[c]["P"]), len(pools[c]["hard_N"])), c))
    b = Builder(seed, classes, present, pools, comp)
    for c in order:
        b.build_class(c)
    rows = to_rows(b.groups, images, present)
    return dict(images=images, classes=classes, present=present, comp=comp, excluded=excluded, rows=rows,
                pilot=pick_pilot(rows), used=used, order=order,
                boundary={classes[c]["name"]: pools[c]["excluded"] for c in used})


def serialize(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the written files")
    args = ap.parse_args()

    bad_files = set()
    for _ in range(3):
        try:
            d = build(extra_drop=bad_files)
        except Shortage as e:
            sys.exit(f"Not enough images: {e}. Nothing written.")
        texts = {f"{s}.jsonl": serialize(d["rows"][s]) for s in SPLITS}
        texts["pilot_train.jsonl"] = serialize(d["pilot"])
        if args.check:
            manifest = json.loads((OUT / "manifest.json").read_text())
            diff = [f for f, t in texts.items() if manifest["outputs"][f]["sha256"] != sha256(t)]
            if diff:
                sys.exit(f"--check: differs from the written files: {diff}")
            print("--check: rebuilt files match manifest.json (same seed, same inputs)")
            return
        used_ids = sorted({i for s in SPLITS for r in d["rows"][s] for i in r["image_ids"]})
        broken = unreadable([COCO / "train2014" / d["images"][i]["file_name"] for i in used_ids])
        if not broken:
            break
        print(f"{len(broken)} unreadable images, rebuilding without them: {broken[:3]}")
        names = {d["images"][i]["file_name"]: i for i in used_ids}
        bad_files |= {names[b.split(":")[0]] for b in broken}
    else:
        sys.exit("images still unreadable after 3 rebuilds")

    comp_sets = {c: {b for b, _, _ in d["comp"][c][0]} for c in d["classes"]}
    checks = run_checks(d["rows"], d["pilot"], d["present"], comp_sets, d["excluded"], d["used"])
    checks["image files open and decode"] = "pass"
    failed = {k: v for k, v in checks.items() if v != "pass"}
    if failed:
        sys.exit(f"Checks failed, nothing written: {json.dumps(failed, indent=1)}")

    stats = statistics(d["rows"], d["pilot"], d["present"])
    stats["checks"] = checks
    texts["statistics.json"] = json.dumps(stats, indent=1) + "\n"
    texts["excluded_images.json"] = json.dumps({
        "global": d["excluded"],
        "per_class_0_to_1pct_or_crowd_only": {k: sorted(v) for k, v in sorted(d["boundary"].items())},
    }) + "\n"
    excl_files = {f"exclude/{p.name}": file_sha256(p) for p in sorted(EXCLUDE_DIR.glob("*.txt"))}
    manifest = {
        "dataset": "COCO train2014 multi-image object existence, 4-case counterfactual groups",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "group": {"cases": {"A": "P1 P2 P3 Yes", "B": "P1 P2 N1 No", "C": "P1 P4 P3 Yes", "D": "P1 P4 N1 No"},
                  "positions": "P1 at fixed_position, P2/P4 at stay_position, P3/N1 at flip_position",
                  "flip_pairs": ["A-B", "C-D"], "stay_pairs": ["A-C", "B-D"],
                  "position_pairs": "all 6 (flip, stay) pairs, equally often per class and split"},
        "question_template": "Is there {a|an} {class_name} in all 3 images?  ('an' before a vowel, as MIHBench)",
        "answer_instruction": "not in the data; the training code adds one if wanted",
        "image_paths": "relative to data/",
        "classes": {"used": len(d["used"]), "build_order": [d["classes"][c]["name"] for c in d["order"]],
                    "left_out": {n: "too few P images (src/coco_pools.py)" for n in EXCLUDED_CLASSES}},
        "params": {"min_area": MIN_AREA, "min_cooccurrence_images": MIN_COOC,
                   "max_companion_share": MAX_COMPANION_SHARE, "hard_top_k": HARD_TOP_K, "hard_N_share": 0.5,
                   "groups_per_class": GROUPS, "groups_per_image": CAP, "reuse_preference": REUSE_PREFERENCE},
        "inputs": {"instances_train2014.json": file_sha256(ANN),
                   "src/coco_pools.py": file_sha256(PROJECT / "src" / "coco_pools.py"),
                   "src/build_coco_existence.py": file_sha256(Path(__file__)), **excl_files},
        "outputs": {f: {"sha256": sha256(t), "lines": t.count("\n")} for f, t in texts.items()
                    if f.endswith(".jsonl")},
        "leakage_checks": {
            "mihbench": {"status": "passed", "overlapping_images": len(d["excluded"]["mihbench"])},
            "covr": {"status": "pending", "how": "put the COCO ids of COVR evaluation images in "
                     "data/coco_existence/exclude/covr.txt and rebuild (same seed)"},
            "splits": "no shared image"},
        "n_caption_filter": "not applied",
        "p_n_visual_audit": "pending",
    }
    texts["manifest.json"] = json.dumps(manifest, indent=1) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    for f, t in texts.items():
        (OUT / f).write_text(t)

    print(f"{'split':<12}{'groups':>7}{'questions':>10}{'images':>8}{'yes':>6}{'no':>6}  configurations")
    for s, v in list(stats["splits"].items()) + [("pilot_train", stats["pilot_train"])]:
        print(f"{s:<12}{v['groups']:>7}{v['questions']:>10}{v['unique_images']:>8}{v['yes']:>6}{v['no']:>6}  "
              f"{v['configurations']}")
    tr = stats["splits"]["train"]
    print(f"train: (flip, stay) pairs {tr['position_pairs_flip_stay']}")
    print(f"train: groups per image {tr['groups_per_image']}, N types {tr['negative_types']}, "
          f"reused images always at the same position {tr['reused_images_always_same_position']}/{tr['reused_images']}")
    print(f"checks: {len(checks)} passed")
    print(f"written: {', '.join(texts)} -> {OUT}")


if __name__ == "__main__":
    main()
