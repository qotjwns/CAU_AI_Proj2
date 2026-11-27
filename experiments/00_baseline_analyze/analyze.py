#!/usr/bin/env python3
"""
    Analysis of 00_baseline_analyze (README.md): reads the raw calls of infer.py, adds the ground truth and writes,
    per model in results/00_baseline_analyze/<model>/ (output.results in config.yaml):
      samples.jsonl      one line per question answered under all of A, B and C (README: samples.jsonl)
      table_subset.csv   results table 1: accuracy of A, B, C per task and overall, A−B and B−C with paired
                         bootstrap 95% CI
      table_metric.csv   results table 2: accuracy, precision, recall, F1, false-yes / false-no rate, yes ratio
      cross.json         per-question A/B/C correctness patterns, image-level errors of A and B, answer counts
    Ground truth follows MIHBench: joint_gt is the question's label. image_gt is [1, 1, 1] for a yes question, the
    COCO val2014 annotation for a no question, or null when COCO has the object in all three images (which image
    MIHBench counts as missing is unknown); label_consistent records whether COCO alone gives the same label.
    Answers are read with the official rule (common/mihbench.parse_yes_no); margins are
    logsumexp(Yes, yes) − logsumexp(No, no) at the first generated token.
    Questions still missing a call are left out, so it can run while infer.py is still going.
    python experiments/00_baseline_analyze/analyze.py                          # every model with results
    python experiments/00_baseline_analyze/analyze.py --model qwen2.5-vl-3b
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from common.mihbench import load_questions, parse_yes_no, question_object  # noqa: E402

CONDITIONS = ("A", "B", "C")
METRICS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1",
           "false_yes_rate": "False-Yes Rate", "false_no_rate": "False-No Rate", "yes_ratio": "Yes Ratio"}


def load_config(path=HERE / "config.yaml"):
    return yaml.safe_load(path.read_text())


def subset_name(task):
    return task.removeprefix("existence_")


# ---- ground truth ----

def coco_objects(path):
    """{COCO val2014 file name: set of the class names annotated in it}."""
    ann = json.loads(Path(path).read_text())
    names = {c["id"]: c["name"] for c in ann["categories"]}
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    objects = defaultdict(set)
    for a in ann["annotations"]:
        objects[files[a["image_id"]]].add(names[a["category_id"]])
    return objects


def ground_truth(question, obj, coco):
    """(joint_gt, image_gt, label_consistent) under the README rule: MIHBench's label decides."""
    joint = int(question["label"].lower() == "yes")
    in_coco = [int(obj in coco[f]) for f in question["image_list"]]
    if joint:
        image = [1] * len(in_coco)
    elif all(in_coco):
        image = None
    else:
        image = in_coco
    return joint, image, int(all(in_coco)) == joint


# ---- answers ----

def read_raw(path):
    """Saved calls by (question_id, image_index). Skips a line cut off by a copy taken mid-write."""
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[row["question_id"], row.get("image_index")] = row
    return rows


def logsumexp(a, b):
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def read_answer(row):
    logits = row["logits"]
    yes, no = logsumexp(logits["Yes"], logits["yes"]), logsumexp(logits["No"], logits["no"])
    return {"response": row["response"], "pred": int(parse_yes_no(row["response"]) == "yes"),
            "yes": round(yes, 4), "no": round(no, 4), "margin": round(yes - no, 4),
            "first_is_answer": row["first_token"].strip().lower() in ("yes", "no")}


def per_image(rows, joint_gt):
    """A or B entry of a sample: the answers about each image, and their AND."""
    answers = [read_answer(r) for r in rows]
    pred = int(all(a["pred"] for a in answers))
    return {"raw_responses": [a["response"] for a in answers], "image_preds": [a["pred"] for a in answers],
            "yes_logits": [a["yes"] for a in answers], "no_logits": [a["no"] for a in answers],
            "margins": [a["margin"] for a in answers],
            "first_token_is_answer": [a["first_is_answer"] for a in answers],
            "joint_pred": pred, "correct": pred == joint_gt}


def joint(row, joint_gt):
    """C entry of a sample: the model's own answer to the whole question."""
    a = read_answer(row)
    return {"raw_response": a["response"], "yes_logit": a["yes"], "no_logit": a["no"], "margin": a["margin"],
            "first_token_is_answer": a["first_is_answer"], "joint_pred": a["pred"], "correct": a["pred"] == joint_gt}


def build_samples(model, raw_dir, cfg, coco):
    """Samples of the questions answered under all three conditions, and (done, total) per task."""
    samples, done = [], {}
    for task in cfg["tasks"]:
        questions = load_questions(ROOT / cfg["data"]["questions"].format(task=task))
        raw = {cond: read_raw(raw_dir / cond / f"{task}.jsonl") for cond in CONDITIONS}
        n_done = 0
        for q in questions:
            qid, n = q["question_id"], len(q["image_list"])
            a = [raw["A"].get((qid, i)) for i in range(n)]
            b = [raw["B"].get((qid, i)) for i in range(n)]
            c = raw["C"].get((qid, None))
            if None in a or None in b or c is None:
                continue
            obj = question_object(q["text"])
            joint_gt, image_gt, consistent = ground_truth(q, obj, coco)
            samples.append({"sample_id": f"{subset_name(task)}_{qid:04d}", "subset": subset_name(task),
                            "model": model, "object": obj, "image_ids": q["image_list"], "image_gt": image_gt,
                            "joint_gt": joint_gt, "label_consistent": consistent,
                            "A": per_image(a, joint_gt), "B": per_image(b, joint_gt), "C": joint(c, joint_gt)})
            n_done += 1
        done[task] = (n_done, len(questions))
    return samples, done


# ---- statistics ----

def scores(preds, gts):
    """Question-level metrics with Yes as the positive class (precision, recall, F1 as the official eval.py)."""
    tp = sum(p and g for p, g in zip(preds, gts))
    fp = sum(p and not g for p, g in zip(preds, gts))
    tn = sum(not p and not g for p, g in zip(preds, gts))
    fn = sum(not p and g for p, g in zip(preds, gts))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(preds), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_yes_rate": fp / (fp + tn) if fp + tn else None,
            "false_no_rate": fn / (fn + tp) if fn + tp else None,
            "yes_ratio": (tp + fp) / len(preds)}


def paired_bootstrap(correct, pairs, cfg):
    """Accuracy differences between conditions on the same questions, with a percentile CI from resampling the
    questions (the same resamples for every pair). correct: {cond: 0/1 array}; pairs: [(x, y), ...]."""
    n_boot, level, seed = cfg["n"], cfg["ci"], cfg["seed"]
    rng = np.random.default_rng(seed)
    diffs = {f"{x}-{y}": correct[x].astype(float) - correct[y] for x, y in pairs}
    n = len(next(iter(diffs.values())))
    boot = {k: np.empty(n_boot) for k in diffs}
    for start in range(0, n_boot, 500):
        idx = rng.integers(0, n, size=(min(500, n_boot - start), n))
        for k, d in diffs.items():
            boot[k][start:start + len(idx)] = d[idx].mean(axis=1)
    q = [(1 - level) / 2, 1 - (1 - level) / 2]
    return {k: {"diff": float(d.mean()), "ci_low": float(np.quantile(boot[k], q[0])),
                "ci_high": float(np.quantile(boot[k], q[1]))} for k, d in diffs.items()}


def image_level(samples):
    """Image-level errors of A and B against image_gt (questions with image_gt null are left out)."""
    out = {"n_images": 0}
    pairs = Counter()
    rows = {"A": [], "B": []}
    for s in samples:
        if s["image_gt"] is None:
            continue
        for i, gt in enumerate(s["image_gt"]):
            out["n_images"] += 1
            right = {}
            for cond in ("A", "B"):
                pred = s[cond]["image_preds"][i]
                rows[cond].append((pred, gt))
                right[cond] = pred == gt
            pairs[("both_right" if right["A"] and right["B"] else "A_only_right" if right["A"]
                   else "B_only_right" if right["B"] else "both_wrong")] += 1
    for cond, pg in rows.items():
        if pg:
            m = scores([p for p, _ in pg], [g for _, g in pg])
            out[cond] = {k: m[k] for k in ("accuracy", "false_yes_rate", "false_no_rate", "yes_ratio")}
    out["A_vs_B"] = {k: pairs[k] for k in ("both_right", "A_only_right", "B_only_right", "both_wrong")}
    return out


# ---- per model ----

def analyze(model, cfg, coco):
    out_dir = ROOT / cfg["output"]["results"].format(model=model)
    samples, done = build_samples(model, out_dir / "raw", cfg, coco)
    print(f"\n== {model}: " + ", ".join(f"{subset_name(t)} {d}/{n}" for t, (d, n) in done.items())
          + " questions answered under A, B and C")
    if not samples:
        return
    with (out_dir / "samples.jsonl").open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    groups = {subset_name(t).capitalize(): [s for s in samples if s["subset"] == subset_name(t)] for t in cfg["tasks"]}
    groups = {name: g for name, g in groups.items() if g}
    groups["Overall"] = samples
    table1, table2, cross = [], [], {"model": model, "questions_done": {t: list(d) for t, d in done.items()},
                                     "groups": {}}
    for name, group in groups.items():
        gts = [s["joint_gt"] for s in group]
        correct = {c: np.array([s[c]["correct"] for s in group], dtype=int) for c in CONDITIONS}
        boot = paired_bootstrap(correct, [("A", "B"), ("B", "C")], cfg["analysis"]["bootstrap"])
        m = {c: scores([s[c]["joint_pred"] for s in group], gts) for c in CONDITIONS}
        table1.append({"subset": name, "n": len(group), **{c: m[c]["accuracy"] for c in CONDITIONS},
                       **{f"{k}{part}": boot[k][key] for k in boot
                          for part, key in (("", "diff"), ("_ci_low", "ci_low"), ("_ci_high", "ci_high"))}})
        table2 += [{"subset": name, "metric": label, **{c: m[c][key] for c in CONDITIONS}}
                   for key, label in METRICS.items()]
        cross["groups"][name] = {
            "n": len(group),
            "label_consistent": sum(s["label_consistent"] for s in group),
            "image_gt_null": sum(s["image_gt"] is None for s in group),
            "patterns": dict(sorted(Counter("".join(f"{c}{int(s[c]['correct'])}" for c in CONDITIONS)
                                            for s in group).items())),
            "bootstrap": boot,
            "image_level": image_level(group),
            "first_token_not_answer": {
                c: sum(not x for s in group for x in (s[c]["first_token_is_answer"] if c != "C"
                                                      else [s[c]["first_token_is_answer"]])) for c in CONDITIONS},
        }
    write_csv(out_dir / "table_subset.csv", table1)
    write_csv(out_dir / "table_metric.csv", table2)
    (out_dir / "cross.json").write_text(json.dumps(cross, indent=2, ensure_ascii=False) + "\n")

    print(f"{'subset':<12}{'n':>6}{'A':>8}{'B':>8}{'C':>8}{'A−B':>8}{'B−C':>8}")
    for r in table1:
        print(f"{r['subset']:<12}{r['n']:>6}" + "".join(f"{r[k]:>8.3f}" for k in ("A", "B", "C", "A-B", "B-C")))
    print(f"-> {out_dir}")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{k: ("" if v is None else round(v, 4) if isinstance(v, float) else v)
                           for k, v in r.items()} for r in rows])


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", choices=list(cfg["models"]),
                    help="default: every model in config.yaml that has results")
    args = ap.parse_args()
    models = args.model or [m for m in cfg["models"]
                            if (ROOT / cfg["output"]["results"].format(model=m) / "raw").is_dir()]
    if not models:
        raise SystemExit("no results yet: run infer.py first")
    coco = coco_objects(ROOT / cfg["data"]["coco_annotations"])
    for model in models:
        analyze(model, cfg, coco)


if __name__ == "__main__":
    main()
