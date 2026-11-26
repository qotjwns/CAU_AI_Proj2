#!/usr/bin/env python3
"""
    Our MIHBench existence results next to the numbers the MIHBench paper reports (arXiv 2508.00726, Table 2,
    baseline rows). The paper gives existence only as the mean over the random, popular and adversarial subsets
    (each metric averaged, e.g. its F1 is not the F1 of its mean precision and recall) and names no model sizes,
    so each run is set against its model family's number.
      figures/mihbench/<method>/paper_comparison.png   accuracy, F1, yes-ratio of every run against the paper
      results/mihbench/<method>/paper_comparison.csv   all 5 metrics per run: ours, the paper's, the difference
    Reads results/mihbench/<method>/summary.csv (subset "all"), so run src/eval_mihbench.py first.
    Row names: "our script" is src/infer_mihbench.py, "DAB script" the paper's own qwen2.5vl_eval.py
    (4-bit, eager attention), "+ yes/no" the official yes/no instruction after the question. Our script always
    adds it now; the -noyn folders are its earlier runs without it (InternVL's prompt always had it).
    python src/compare_paper.py
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from eval_mihbench import AXIS, GRID, INK, INK_2, MODEL_LABELS, SURFACE, assign_colors, split_variant

PROJECT = Path(__file__).resolve().parents[1]
EXISTENCE = ("existence_random", "existence_popular", "existence_adversarial")
KEYS = ("accuracy", "precision", "recall", "f1", "yes_ratio")
# MIHBench paper, Table 2, Existence, baseline rows, in %: accuracy, precision, recall, F1, yes ratio.
PAPER = {
    "Qwen2.5-VL": (71.59, 88.69, 51.67, 64.70, 30.08),
    "InternVL2.5": (74.00, 88.29, 57.42, 69.16, 33.42),
    "LLaVA-NeXT-Interleave": (75.75, 87.97, 62.92, 72.68, 37.17),
}
FAMILIES = (("qwen2.5-vl", "Qwen2.5-VL"), ("internvl2.5", "InternVL2.5"),
            ("llava", "LLaVA-NeXT-Interleave"))  # our LLaVA-OV-7B stands in for LLaVA-NeXT-Interleave
PLOTTED = (("accuracy", "Accuracy"), ("f1", "F1"), ("yes_ratio", "Yes ratio"))


def family(model):
    return next((f for prefix, f in FAMILIES if model.startswith(prefix)), None)


def run_label(model):
    """'qwen2.5-vl-7b-official-yn' -> 'Qwen2.5-VL-7B · DAB script + yes/no'."""
    base, variant = split_variant(model)
    tokens = variant.split("-") if variant else []
    extras = [x for t, x in (("4bit", "4-bit"), ("sdpa", "sdpa")) if t in tokens]
    official = "official" in tokens
    yes_no = "yn" in tokens if official else "noyn" not in tokens  # ours: always, except the -noyn runs
    label = f"{MODEL_LABELS.get(base, base)} · {'DAB' if official else 'our'} script"
    return label + (f" ({', '.join(extras)})" if extras else "") + (" + yes/no" if yes_no else "")


def load_runs(summary):
    """[(model, family, {metric: mean over the 3 existence subsets, in %})] in summary.csv order."""
    per = {}
    with open(summary) as f:
        for r in csv.DictReader(f):
            if r["subset"] == "all" and r["task"] in EXISTENCE:
                per.setdefault(r["model"], {})[r["task"]] = r
    runs = []
    for model, tasks in per.items():
        fam = family(model)
        if fam is None or len(tasks) < len(EXISTENCE):
            print(f"skipped {model}: " + ("no paper number" if fam is None else "not all 3 existence subsets"))
            continue
        runs.append((model, fam, {k: sum(100 * float(tasks[t][k]) for t in EXISTENCE) / len(EXISTENCE)
                                  for k in KEYS}))
    return runs


def plot(runs, out_path):
    families = [f for _, f in FAMILIES if any(r[1] == f for r in runs)]
    colors = assign_colors([m for m, _, _ in runs])
    y, ticks, labels, heads, row_y, span = 0.0, [], [], set(), {}, {}
    for fam in families:  # a header row per family, then its runs
        ticks.append(y), labels.append(fam), heads.add(len(labels) - 1)
        y += 1
        start = y
        for m, f, _ in runs:
            if f == fam:
                row_y[m] = y
                ticks.append(y), labels.append(run_label(m))
                y += 1
        span[fam] = (start - 0.35, y - 0.65)
        y += 0.5  # air between families

    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9})
    fig, axes = plt.subplots(1, len(PLOTTED), figsize=(11, 0.3 * y + 1.5), dpi=200, sharey=True,
                             facecolor=SURFACE)
    for ax, (key, title) in zip(axes, PLOTTED):
        k = KEYS.index(key)
        ax.set_facecolor(SURFACE)
        ax.set_xlim(0, 100)
        ax.set_ylim(y - 0.4, -0.6)
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"])
        ax.set_yticks(ticks, labels)
        ax.tick_params(colors=INK_2, length=0, labelsize=8.5)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        if key == "yes_ratio":  # half the labels are yes
            ax.axvline(50, color=INK_2, linewidth=0.8, zorder=1)
            ax.text(51, y - 0.6, "label yes share", color=INK_2, fontsize=7.5, va="bottom")
        for fam in families:
            p = PAPER[fam][k]
            ax.vlines(p, *span[fam], color=INK, linewidth=2, zorder=2)
            ax.text(p, span[fam][0] - 0.15, f"{p:.1f}", color=INK, fontsize=7.5, ha="center", va="bottom")
        for m, fam, ours in runs:
            p, v = PAPER[fam][k], ours[key]
            ax.hlines(row_y[m], min(p, v), max(p, v), color=AXIS, linewidth=2, zorder=1)
            ax.plot(v, row_y[m], "o", markersize=7, color=colors[m], markeredgecolor=SURFACE,
                    markeredgewidth=1.2, zorder=3)
    for i, t in enumerate(axes[0].get_yticklabels()):
        if i in heads:
            t.set_fontweight("bold"), t.set_color(INK)
    handles = [Line2D([], [], color=INK, marker="|", markersize=11, markeredgewidth=2, linestyle="none"),
               Line2D([], [], color=INK_2, marker="o", markersize=7, linestyle="none")]
    fig.legend(handles, ["paper (MIHBench Table 2, baseline)", "our run"], loc="upper center", ncol=2,
               frameon=False, fontsize=8.5, labelcolor=INK_2, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "MIHBench existence, each metric averaged over random, popular and adversarial "
             "(800 questions each). The paper names no model sizes; LLaVA-OV-7B stands in for "
             "LLaVA-NeXT-Interleave.", ha="center", va="bottom", color=INK_2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="vanilla")
    ap.add_argument("--results-dir", type=Path, default=PROJECT / "results" / "mihbench")
    ap.add_argument("--figures-dir", type=Path, default=PROJECT / "figures" / "mihbench")
    args = ap.parse_args()
    summary = args.results_dir / args.method / "summary.csv"
    if not summary.exists():
        raise SystemExit(f"{summary} not found: python src/eval_mihbench.py --method {args.method}")
    runs = load_runs(summary)
    if not runs:
        raise SystemExit("no run with all 3 existence subsets")

    out_csv = args.results_dir / args.method / "paper_comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "paper_model"] + [f"{s}_{k}" for k in KEYS for s in ("ours", "paper", "diff")])
        for m, fam, ours in runs:
            w.writerow([m, fam] + [f"{x:.2f}" for i, k in enumerate(KEYS)
                                   for x in (ours[k], PAPER[fam][i], ours[k] - PAPER[fam][i])])
    print(f"{'run':<44}{'paper model':<24}" + "".join(f"{h:>20}" for h in ("acc ours/paper", "f1", "yes")))
    for m, fam, ours in runs:
        cells = [f"{ours[k]:5.1f}/{PAPER[fam][KEYS.index(k)]:5.1f} ({ours[k] - PAPER[fam][KEYS.index(k)]:+5.1f})"
                 for k in ("accuracy", "f1", "yes_ratio")]
        print(f"{run_label(m):<44}{fam:<24}" + "".join(f"{c:>20}" for c in cells))
    out_png = args.figures_dir / args.method / "paper_comparison.png"
    plot(runs, out_png)
    print(f"\nTable: {out_csv}\nFigure: {out_png}")


if __name__ == "__main__":
    main()
