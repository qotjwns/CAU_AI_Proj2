#!/usr/bin/env python3
"""
    Score MIHBench answers and plot them, from results/mihbench/<method>/<model>/<task>.jsonl
    (written by src/infer_mihbench.py). Every answer is re-parsed with the official yes/no rule.
      results/mihbench/<method>/summary.csv              one row per model x task x subset
      figures/mihbench/<method>/mihbench.png             accuracy, F1, yes-ratio per model and task
      figures/mihbench/<method>/mihbench_<subset>.png      the same for every other plotted subset
    subset "all" is every question; "no_typo" drops the 86 existence questions whose text is broken
    ("Is there an umbrella in the imange? in all 3 images?"); "seen_classes" drops the questions on classes left
    out of the COCO training data (toaster, hair drier: EXCLUDED_CLASSES in src/coco_pools.py) and
    "unseen_classes" keeps only those (58 existence, 7 count questions; printed, not plotted).
    Existence only, judged with the COCO val2014 annotations (skipped without instances_val2014.json):
    "label_consistent" keeps the questions whose label agrees with the annotations (yes exactly when every image
    holds the object; about 40% of the yes labels do not), "large_objects" those whose images all fit the
    training P/N definition (the object absent, or at >= 1% of the image).
    python src/eval_mihbench.py                    # method vanilla
    python src/eval_mihbench.py --method vanilla --results-dir results/mihbench
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath

from coco_pools import EXCLUDED_CLASSES, mihbench_existence_flags
from mihbench import TASKS, parse_yes_no, score

PROJECT = Path(__file__).resolve().parents[1]
N_QUESTIONS = 800
UNSEEN = re.compile(r"\b(?:" + "|".join(map(re.escape, EXCLUDED_CLASSES)) + r")s?\b")  # "toaster", "toasters"
SUBSETS = {  # subset -> figure footnote (None: not plotted)
    "all": "every question",
    "no_typo": 'without the 86 "in the imange?" questions',
    "seen_classes": f"without the questions on classes left out of training ({', '.join(EXCLUDED_CLASSES)})",
    "unseen_classes": None,
    "label_consistent": "existence questions whose label agrees with the COCO annotations",
    "large_objects": "existence questions whose images hold the object at >= 1% of the image or not at all",
}
TASK_LABELS = {
    "existence_random": "Random",
    "existence_popular": "Popular",
    "existence_adversarial": "Adversarial",
    "count": "Count",
    "id_consistency": "ID consistency",
}
# Fixed color per model, so a model keeps its color whichever others are plotted (validated categorical slots).
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
MODEL_ORDER = ["qwen2.5-vl-3b", "qwen2.5-vl-7b", "internvl2.5-4b", "llava-ov-7b"]
MODEL_LABELS = {"qwen2.5-vl-3b": "Qwen2.5-VL-3B", "qwen2.5-vl-7b": "Qwen2.5-VL-7B",
                "internvl2.5-4b": "InternVL2.5-4B", "llava-ov-7b": "LLaVA-OV-7B"}
INK, INK_2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"
METRICS = [("accuracy", "Accuracy"), ("f1", "F1"), ("yes_ratio", "Yes ratio")]


def split_variant(name):
    """'internvl2.5-4b-sdpa' -> ('internvl2.5-4b', 'sdpa')."""
    for base in MODEL_ORDER:
        if name == base or name.startswith(base + "-"):
            return base, name[len(base) + 1:]
    return name, ""


def display_name(name):
    base, variant = split_variant(name)
    return MODEL_LABELS.get(base, base) + (f" ({variant})" if variant else "")


def assign_colors(models):
    """Known base models take their own slot; a second variant of the same base or an unknown model takes the
    next free slot."""
    colors, used = {}, set()
    for m in models:
        base, _ = split_variant(m)
        slot = MODEL_ORDER.index(base) if base in MODEL_ORDER else None
        if slot is None or slot in used:
            free = [i for i in range(len(MODEL_ORDER), len(SERIES)) if i not in used]
            if not free:
                raise SystemExit(f"too many models for one figure ({len(models)}); plot them in groups")
            slot = free[0]
        used.add(slot)
        colors[m] = SERIES[slot]
    return colors


def load(method_dir):
    """{model: {task: rows}} for every <model>/<task>.jsonl, models in a fixed order."""
    found = {}
    for model_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
        tasks = {t: [json.loads(line) for line in open(model_dir / f"{t}.jsonl") if line.strip()]
                 for t in TASKS if (model_dir / f"{t}.jsonl").exists()}
        if tasks:
            found[model_dir.name] = tasks
    key = lambda m: (MODEL_ORDER.index(split_variant(m)[0]) if split_variant(m)[0] in MODEL_ORDER else 99, m)
    return {m: found[m] for m in sorted(found, key=key)}


def summarize(data, method, flags):
    rows = []
    for model, tasks in data.items():
        for task, answers in tasks.items():
            if len(answers) != N_QUESTIONS:
                print(f"warning: {model}/{task} has {len(answers)} answers, not {N_QUESTIONS}")
            for a in answers:
                a["pred"] = parse_yes_no(a["text"])
            subsets = {"all": answers, "no_typo": [a for a in answers if "imange" not in a["question"]],
                       "seen_classes": [a for a in answers if not UNSEEN.search(a["question"])],
                       "unseen_classes": [a for a in answers if UNSEEN.search(a["question"])]}
            if flags is not None and task.startswith("existence_"):
                subsets["label_consistent"] = [a for a in answers if flags[task, a["question_id"]][0]]
                subsets["large_objects"] = [a for a in answers if flags[task, a["question_id"]][1]]
            for subset, subset_rows in subsets.items():
                if subset_rows:  # id_consistency has no unseen-class questions
                    rows.append({"method": method, "model": model, "task": task, "subset": subset,
                                 **score(subset_rows),
                                 "label_yes_ratio": sum(a["label"].lower() == "yes" for a in subset_rows)
                                 / len(subset_rows)})
    return rows


def rounded_bar(ax, x, width, height, color, radius_pt=2.5):
    """A bar from y=0 with rounded corners at its top end only (square at the baseline), drawn as one path."""
    if height <= 0:
        return
    bbox = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    r_px = radius_pt * ax.figure.dpi / 72
    rx = min(r_px * (x1 - x0) / bbox.width, width / 2)  # the radius in x and in y data units
    ry = min(r_px * (y1 - y0) / bbox.height, height)
    k = 0.5523  # cubic Bezier handle length for a quarter circle
    verts = [(x, 0), (x, height - ry),
             (x, height - ry + k * ry), (x + rx - k * rx, height), (x + rx, height),
             (x + width - rx, height),
             (x + width - rx + k * rx, height), (x + width, height - ry + k * ry), (x + width, height - ry),
             (x + width, 0), (x, 0)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, linewidth=0))


def plot(rows, subset, out_path):
    rows = [r for r in rows if r["subset"] == subset]
    models = list(dict.fromkeys(r["model"] for r in rows))
    tasks = [t for t in TASKS if any(r["task"] == t for r in rows)]
    colors = assign_colors(models)
    value = {(r["model"], r["task"]): r for r in rows}

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.4 * len(METRICS), 3.2), dpi=200, facecolor=SURFACE)
    group = 0.76  # share of each task slot taken by its bars; the rest is air between tasks
    gap = 0.012   # surface gap between touching bars, in slot units
    width = group / len(models)
    for ax, (metric, title) in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        ax.set_xlim(-0.5, len(tasks) - 0.5)
        ax.set_ylim(0, 100)
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="semibold", pad=8)
        ax.set_yticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
        ax.set_xticks(range(len(tasks)), [TASK_LABELS[t] for t in tasks])
        ax.tick_params(colors=INK_2, length=0, labelsize=8.5)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        fig.canvas.draw()  # fixes the axes size, which rounded_bar needs to convert points to data units
        for i, task in enumerate(tasks):
            for j, model in enumerate(models):
                r = value.get((model, task))
                if r is None:
                    continue
                x = i - group / 2 + j * width + gap / 2
                rounded_bar(ax, x, width - gap, 100 * r[metric], colors[model])
        if metric == "yes_ratio":  # an unbiased model says yes as often as the labels do: 50% in the full set,
            for i, task in enumerate(tasks):  # less in label_consistent / large_objects, so mark it per task
                share = 100 * next(r["label_yes_ratio"] for r in rows if r["task"] == task)
                ax.hlines(share, i - group / 2, i + group / 2, color=INK_2, linewidth=0.8)
            ax.text(len(tasks) - 0.5, share + 1.5, "label yes share", color=INK_2, fontsize=7.5,
                    ha="right", va="bottom")
    handles = [Rectangle((0, 0), 1, 1, facecolor=colors[m], linewidth=0) for m in models]
    fig.legend(handles, [display_name(m) for m in models], loc="upper center", ncol=len(models), frameon=False,
               fontsize=8.5, labelcolor=INK_2, handlelength=0.9, handleheight=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, f"MIHBench, {SUBSETS[subset]}", ha="center", va="bottom", color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="vanilla")
    ap.add_argument("--results-dir", type=Path, default=PROJECT / "results" / "mihbench")
    ap.add_argument("--figures-dir", type=Path, default=PROJECT / "figures" / "mihbench")
    args = ap.parse_args()
    method_dir = args.results_dir / args.method
    if not method_dir.is_dir():
        raise SystemExit(f"{method_dir} not found")
    data = load(method_dir)
    if not data:
        raise SystemExit(f"no <model>/<task>.jsonl under {method_dir}")
    flags = mihbench_existence_flags()
    if flags is None:
        print("warning: data/coco/annotations/instances_val2014.json not found, "
              "skipping label_consistent and large_objects (python scripts/download_coco_train2014.py)")
    rows = summarize(data, args.method, flags)

    summary = method_dir / "summary.csv"
    with open(summary, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for subset, note in SUBSETS.items():
        if not any(r["subset"] == subset for r in rows):
            continue
        print(f"\n[{subset}]")
        print(f"{'model':<24}{'task':<24}{'n':>5}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}{'yes':>8}")
        for r in rows:
            if r["subset"] == subset:
                print(f"{r['model']:<24}{r['task']:<24}{r['n']:>5}" + "".join(
                    f"{r[k]:>8.3f}" for k in ("accuracy", "precision", "recall", "f1", "yes_ratio")))
        if note:
            suffix = "" if subset == "all" else f"_{subset}"
            plot(rows, subset, args.figures_dir / args.method / f"mihbench{suffix}.png")
    print(f"\nSummary: {summary}\nFigures: {args.figures_dir / args.method}")


if __name__ == "__main__":
    main()
