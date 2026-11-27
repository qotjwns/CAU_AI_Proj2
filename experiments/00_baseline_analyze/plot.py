#!/usr/bin/env python3
"""
    Figures of 00_baseline_analyze from analyze.py's tables (results/00_baseline_analyze/<model>/table_*.csv),
    one panel per model, written to figures/00_baseline_analyze/:
      accuracy.png      results table 1: accuracy of A, B, C per subset (dots on one row per subset)
      differences.png   A−B (multi-image context) and B−C (evidence integration) with paired bootstrap 95% CI
      error_rates.png   results table 2: question-level false-yes and false-no rates of A, B, C
    Colors: A/B/C and A−B/B−C are two categorical sets, each checked for color-blind separation; the values
    themselves are in the CSV tables.
    python experiments/00_baseline_analyze/plot.py
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "figures" / "00_baseline_analyze"

COND = {"A": ("#257de0", "A  each image alone, AND"),
        "B": ("#f47230", "B  3 images, asked per image, AND"),
        "C": ("#00b894", "C  official question")}
DIFF = {"A-B": ("#6a4ce2", "A−B  multi-image context"), "B-C": ("#008a17", "B−C  evidence integration")}
SUBSETS = ["Random", "Popular", "Adversarial", "Overall"]
INK, INK2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})


def read(model, name):
    path = ROOT / "results" / "00_baseline_analyze" / model / name
    return list(csv.DictReader(path.open())) if path.exists() else None


def style(ax, grid_axis):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def legend(fig, items, whisker=False):
    handles = [plt.Line2D([], [], color=c, marker="o", linestyle="-" if whisker else "none",
                          markersize=7, markeredgecolor=SURFACE, linewidth=2) for c, _ in items]
    fig.legend(handles, [label for _, label in items], loc="upper center", ncol=len(items), frameon=False,
               bbox_to_anchor=(0.5, 0.95), handletextpad=0.4, columnspacing=1.6)


def accuracy(models, tables):
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.4), sharex=True, sharey=True)
    for ax, model in zip(axes, models):
        rows = {r["subset"]: r for r in tables[model]}
        for y, subset in enumerate(reversed(SUBSETS)):
            vals = {c: float(rows[subset][c]) for c in COND}
            ax.plot([min(vals.values()), max(vals.values())], [y, y], color=AXIS, linewidth=1.5, zorder=1)
            offsets = dict(zip(COND, (0.14, 0, -0.14)))
            for c, (color, _) in COND.items():
                ax.plot(vals[c], y + offsets[c], "o", color=color, markersize=8, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, zorder=3)
            if subset == "Overall":  # direct labels at the two ends of the row only, level with their own dot
                lo, hi = min(vals, key=vals.get), max(vals, key=vals.get)
                ax.text(vals[lo] - 0.008, y + offsets[lo], f"{lo} {vals[lo]:.3f}", ha="right", va="center",
                        color=INK, fontsize=8)
                ax.text(vals[hi] + 0.008, y + offsets[hi], f"{hi} {vals[hi]:.3f}", ha="left", va="center",
                        color=INK, fontsize=8)
        ax.set_yticks(range(len(SUBSETS)), list(reversed(SUBSETS)))
        ax.set_title(model, color=INK, fontsize=10, loc="left")
        ax.set_xlim(0.52, 0.90)
        ax.set_xlabel("Accuracy (MIHBench label)")
        style(ax, "x")
    fig.suptitle("Accuracy by condition", x=0.01, ha="left", fontsize=12, color=INK)
    legend(fig, list(COND.values()))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def differences(models, tables):
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.4), sharex=True, sharey=True)
    for ax, model in zip(axes, models):
        rows = {r["subset"]: r for r in tables[model]}
        ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
        for y, subset in enumerate(reversed(SUBSETS)):
            for offset, (k, (color, _)) in zip((0.13, -0.13), DIFF.items()):
                r = rows[subset]
                d, lo, hi = float(r[k]), float(r[f"{k}_ci_low"]), float(r[f"{k}_ci_high"])
                ax.plot([lo, hi], [y + offset] * 2, color=color, linewidth=2, solid_capstyle="round", zorder=2)
                ax.plot(d, y + offset, "o", color=color, markersize=7, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, zorder=3)
                if subset == "Overall":
                    ax.text(hi + 0.008, y + offset, f"{d:+.3f}", ha="left", va="center", color=INK, fontsize=8)
        ax.set_yticks(range(len(SUBSETS)), list(reversed(SUBSETS)))
        ax.set_title(model, color=INK, fontsize=10, loc="left")
        ax.set_xlim(-0.08, 0.25)
        ax.set_xlabel("Accuracy difference (95% CI)")
        style(ax, "x")
    fig.suptitle("Where accuracy is lost: A−B and B−C", x=0.01, ha="left", fontsize=12, color=INK)
    legend(fig, list(DIFF.values()), whisker=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def error_rates(models, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (metric, story, top) in zip(axes, (("False-Yes Rate", "B", 0.2), ("False-No Rate", "C", 0.75))):
        width = 0.24
        for i, c in enumerate(COND):
            xs = [m + (i - 1) * width for m in range(len(models))]
            vals = [float(next(r[c] for r in metrics[m] if r["subset"] == "Overall" and r["metric"] == metric))
                    for m in models]
            ax.bar(xs, vals, width, color=COND[c][0], edgecolor=SURFACE, linewidth=2, zorder=2)
            if c == story:  # label the condition the panel is about
                for x, v in zip(xs, vals):
                    ax.text(x, v + top * 0.015, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=8)
        ax.set_xticks(range(len(models)), models)
        ax.set_ylim(0, top)
        ax.set_title(f"{metric} (overall, question level)", color=INK, fontsize=10, loc="left")
        style(ax, "y")
        ax.spines["bottom"].set_color(AXIS)
    fig.suptitle("Error types: false-yes = says Yes on a No question, false-no = says No on a Yes question",
                 x=0.01, ha="left", fontsize=11, color=INK)
    legend(fig, list(COND.values()))
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    return fig


def main():
    cfg = yaml.safe_load((HERE / "config.yaml").read_text())
    models = [m for m in cfg["models"] if read(m, "table_subset.csv")]
    if not models:
        sys.exit("no tables yet: run analyze.py first")
    subsets = {m: read(m, "table_subset.csv") for m in models}
    metrics = {m: read(m, "table_metric.csv") for m in models}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fig in (("accuracy", accuracy(models, subsets)), ("differences", differences(models, subsets)),
                      ("error_rates", error_rates(models, metrics))):
        fig.savefig(OUT / f"{name}.png", dpi=200)
        plt.close(fig)
        print(OUT / f"{name}.png")


if __name__ == "__main__":
    main()
