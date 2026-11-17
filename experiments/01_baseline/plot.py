"""
baseline 결과 그림을 outputs/figures/01_baseline/<실행 이름>/에 저장 (전체 실행 결과용)
python experiments/01_baseline/plot.py qwen2.5-vl-3b                          # 그림 3장
python experiments/01_baseline/plot.py qwen2.5-vl-3b qwen2.5-vl-3b_float32    # + 두 실행의 margin 비교 그림
그림의 숫자는 mivlm.evaluation.metrics의 결과를 그대로 그린다. 표로 보려면 summarize.py를 쓴다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from mivlm.config import PREDICTIONS_DIR, PROJECT_ROOT
from mivlm.evaluation.metrics import BOUNDARY_MARGIN, VARIANT_ORDER, load_rows, summarize_by

EXPERIMENT = "01_baseline"

# 색 (밝은 배경 기준)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"  # 강조하지 않는 표시
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"  # 강조색
# 값이 클수록 진해지는 한 가지 색의 단계
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK_SECONDARY, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK_SECONDARY, "ytick.labelcolor": INK_SECONDARY,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 13, "axes.titleweight": "bold",
}


def set_title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9)


def save(fig, out_path: Path) -> None:
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {out_path}")


def plot_yes_rate_by_variant(rows: list[dict], run: str, out_path: Path) -> None:
    """변형마다 Yes로 답한 비율. PPP만 정답이 Yes이고, 나머지에서는 이 비율이 False-Yes rate다"""
    by_variant = summarize_by(rows, "variant")
    rates = [by_variant[variant]["yes_rate"] for variant in VARIANT_ORDER]
    colors = [MUTED if variant == "PPP" else BLUE for variant in VARIANT_ORDER]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(VARIANT_ORDER, rates, width=0.3, color=colors)
    ax.bar_label(bars, labels=[f"{rate:.1%}" for rate in rates], padding=3, color=INK)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y")
    ax.tick_params(length=0)
    ax.set_xlabel("images in slots 1-3 (P: object present, N: object absent)")
    ax.set_ylabel("share of 'Yes' answers")
    ax.legend(handles=[Patch(color=MUTED, label="correct answer is Yes"),
                       Patch(color=BLUE, label="correct answer is No (False-Yes)")], frameon=False)
    set_title(ax, "'Yes' answers by variant", f"{run}, {by_variant['PPP']['n']} cases per variant")
    save(fig, out_path)


def plot_margin_by_variant(rows: list[dict], run: str, out_path: Path) -> None:
    """변형마다 logit margin의 분포. 0의 오른쪽이면 Yes, 왼쪽이면 No가 더 그럴듯하다는 뜻"""
    by_variant = summarize_by(rows, "variant")
    margins = {variant: [row["logit_margin"] for row in rows if row["variant"] == variant] for variant in VARIANT_ORDER}
    lowest = min(row["logit_margin"] for row in rows)
    highest = max(row["logit_margin"] for row in rows)
    # bf16 margin은 0.125의 배수다. 값이 구간 경계에 걸리지 않게 경계를 0.1875 + 0.375k에 둔다.
    # 그러면 구간 하나에 값이 정확히 3개씩 들어가고, 가운데 구간은 -0.125, 0, +0.125를 담는다
    edges = [0.1875 + 0.375 * k for k in range(-60, 60)]

    fig, axes = plt.subplots(len(VARIANT_ORDER), 1, figsize=(7, 8.5), sharex=True, sharey=True)
    for ax, variant in zip(axes, VARIANT_ORDER):
        ax.axvspan(-BOUNDARY_MARGIN, BOUNDARY_MARGIN, color=GRID, linewidth=0)
        ax.hist(margins[variant], bins=edges, color=BLUE, rwidth=0.85)
        ax.axvline(0, color=MUTED, linewidth=1)
        ax.grid(axis="y")
        ax.tick_params(length=0)
        summary = by_variant[variant]
        ax.text(0.99, 0.80, variant, transform=ax.transAxes, ha="right", fontweight="bold")
        ax.text(0.99, 0.62, f"mean {summary['mean_margin']:+.2f}\n'Yes' {summary['yes_rate']:.1%}",
                transform=ax.transAxes, ha="right", va="top", color=INK_SECONDARY, fontsize=9)
    axes[0].set_xlim(lowest - 0.5, highest + 0.5)
    axes[-1].set_xlabel("logit margin = Yes logit - No logit   (below 0: No more likely, above 0: Yes more likely)")
    fig.supylabel("cases", color=INK_SECONDARY, fontsize=10)
    set_title(axes[0], "Logit margin by variant", f"{run}, shaded band: |margin| <= {BOUNDARY_MARGIN}")
    save(fig, out_path)


def plot_yes_rate_by_class(rows: list[dict], run: str, out_path: Path) -> None:
    """클래스 x 변형마다 Yes로 답한 비율 (칸 하나에 케이스 20개). 같은 표를 summarize.py가 숫자로 출력한다"""
    by_class = {name: summarize_by([row for row in rows if row["object"] == name], "variant")
                for name in summarize_by(rows, "object")}
    # N이 1장인 세 변형의 Yes 비율이 높은 클래스를 위에 둔다
    names = sorted(by_class, key=lambda name: -sum(by_class[name][variant]["yes_rate"] for variant in ["PPN", "PNP", "NPP"]))
    matrix = [[by_class[name][variant]["yes_rate"] for variant in VARIANT_ORDER] for name in names]

    fig, ax = plt.subplots(figsize=(6.2, 7.5))
    # 칸 사이를 배경색 선으로 띄운다
    mesh = ax.pcolormesh(matrix, cmap=LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP), vmin=0, vmax=1,
                         edgecolors=SURFACE, linewidth=2)
    ax.invert_yaxis()
    ax.set_xticks([i + 0.5 for i in range(len(VARIANT_ORDER))], ["PPP\n(Yes is correct)"] + VARIANT_ORDER[1:])
    ax.set_yticks([i + 0.5 for i in range(len(names))], names)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(mesh, ax=ax, format=PercentFormatter(1.0), shrink=0.6)
    colorbar.set_label("share of 'Yes' answers")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0)
    n_per_cell = by_class[names[0]]["PPP"]["n"]
    set_title(ax, "'Yes' answers by class and variant", f"{run}, {n_per_cell} cases per cell, classes sorted by PPN + PNP + NPP")
    save(fig, out_path)


def plot_margin_comparison(rows: list[dict], other_rows: list[dict], run: str, other_run: str, out_path: Path) -> None:
    """같은 케이스의 margin을 두 실행에서 비교한다. 답이 바뀐 케이스만 강조색"""
    other_by_id = {row["case_id"]: row for row in other_rows}
    same, flipped = [], []
    for row in rows:
        other = other_by_id[row["case_id"]]
        point = (row["logit_margin"], other["logit_margin"])
        (same if row["prediction"] == other["prediction"] else flipped).append(point)
    values = [value for point in same + flipped for value in point]
    full_range = (min(values) - 0.5, max(values) + 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, limits, title in zip(axes, [full_range, (-1.5, 1.5)], ["All cases", "Zoom near 0"]):
        ax.axhline(0, color=GRID, linewidth=1)
        ax.axvline(0, color=GRID, linewidth=1)
        ax.axline((0, 0), slope=1, color=AXIS, linewidth=1)
        ax.scatter([x for x, _ in same], [y for _, y in same], s=8, color=MUTED, alpha=0.35, linewidths=0,
                   label=f"same answer ({len(same):,})")
        ax.scatter([x for x, _ in flipped], [y for _, y in flipped], s=40, color=BLUE, edgecolors=SURFACE, linewidths=1.5,
                   label=f"answer differs ({len(flipped):,})")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect("equal")
        ax.tick_params(length=0)
        ax.set_xlabel(f"logit margin, {run}")
        ax.set_ylabel(f"logit margin, {other_run}")
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Logit margin per case: {run} vs {other_run}", x=0.07, ha="left", fontsize=13, fontweight="bold")
    save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="baseline 결과 그림")
    parser.add_argument("run", help="outputs/predictions/01_baseline/ 아래의 실행 폴더 이름")
    parser.add_argument("compare_with", nargs="?", help="margin을 비교할 두 번째 실행 폴더 이름")
    args = parser.parse_args()

    plt.rcParams.update(STYLE)
    rows = load_rows(PREDICTIONS_DIR / EXPERIMENT / args.run)
    out_dir = PROJECT_ROOT / "outputs" / "figures" / EXPERIMENT / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_yes_rate_by_variant(rows, args.run, out_dir / "yes_rate_by_variant.png")
    plot_margin_by_variant(rows, args.run, out_dir / "margin_by_variant.png")
    plot_yes_rate_by_class(rows, args.run, out_dir / "yes_rate_by_class.png")
    if args.compare_with:
        other_rows = load_rows(PREDICTIONS_DIR / EXPERIMENT / args.compare_with)
        plot_margin_comparison(rows, other_rows, args.run, args.compare_with, out_dir / f"margin_vs_{args.compare_with}.png")


if __name__ == "__main__":
    main()
