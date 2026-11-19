"""
클래스 내부 민감도 분석의 그림을 outputs/figures/01_baseline/<실행 이름>/에 저장
python experiments/01_baseline/plot_class_sensitivity.py qwen2.5-vl-3b

분석의 그림은 4장이다. 첫 장(클래스별 Yes 비율 히트맵)은 plot.py가 이미 그리는 yes_rate_by_class.png를 그대로 쓰고,
이 스크립트는 나머지 3장을 그린다. 값은 class_sensitivity.py의 표와 같은 함수로 계산한다.
  class_margin_drop.png              클래스마다, 변형마다의 짝지은 margin 하락
  class_tendency_vs_margin_drop.png  클래스의 PPP margin(성향)과 margin 하락의 관계
  class_within_auroc.png             클래스 내부 AUROC와, 클래스를 섞어 한 번에 계산한 AUROC
클래스는 세 그림 모두 PPP의 평균 margin이 높은 순서로 놓는다. 구간과 범위의 뜻은 mivlm.analysis.class_sensitivity 참고.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import PercentFormatter

from mivlm.analysis.class_sensitivity import (
    NO_DIFFERENCE, ONE_N, class_pairs, class_values, leave_one_image_out, mean_interval,
)
from mivlm.config import FIGURES_DIR, PREDICTIONS_DIR
from mivlm.evaluation.metrics import ONE_N_VARIANTS, load_rows
from mivlm.plotting import AQUA, AXIS, BLUE, BLUE_RAMP, GRID, INK_SECONDARY, MUTED, ORANGE, STYLE, save, set_title

EXPERIMENT = "01_baseline"
N_SLOT = {"PPN": 3, "PNP": 2, "NPP": 1}  # 변형마다 N이 들어간 자리


def style_class_axis(ax, names: list[str], summary_y: float) -> None:
    """세로축에 클래스를 위에서 아래로 놓고, 선으로 띄운 맨 아래 줄에 클래스 평균을 둔다"""
    ax.set_yticks([*range(len(names)), summary_y], [*names, f"mean of {len(names)} classes\n(95% interval)"])
    ax.set_ylim(summary_y + 0.8, -0.8)  # 위에서 아래로 읽도록 뒤집는다
    ax.axhline(len(names) - 0.25, color=AXIS, linewidth=0.8)
    ax.grid(axis="x")
    ax.tick_params(length=0)


def draw_class_mean(ax, per_class: list[float], y: float, color: str) -> None:
    """클래스 평균을 점으로, 클래스 단위 bootstrap 95% 구간을 가로 막대로 그린다"""
    mean, low, high = mean_interval(per_class)
    ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", color=color, markersize=6, linewidth=2, zorder=3)


def plot_margin_drop(table: dict[str, dict], subtitle: str, out_path: Path) -> None:
    """클래스마다, 변형마다: 같은 pair 안에서 PPP의 margin - N을 넣은 변형의 margin의 평균"""
    names = list(table)
    summary_y = len(names) + 0.5
    fig, ax = plt.subplots(figsize=(8, 8.4))
    # 색은 N의 자리를 따른다 (probe 그림과 같게: 자리 1 파랑, 자리 2 주황, 자리 3 초록)
    for variant, color, offset in zip(ONE_N_VARIANTS, [AQUA, ORANGE, BLUE], [-0.24, 0, 0.24]):
        drops = [table[name][f"margin_drop_{variant}"] for name in names]
        ax.scatter(drops, [y + offset for y in range(len(names))], s=34, color=color, zorder=3,
                   label=f"{variant} (N in slot {N_SLOT[variant]})")
        draw_class_mean(ax, drops, summary_y + offset, color)
    ax.axvline(0, color=MUTED, linewidth=1)
    style_class_axis(ax, names, summary_y)
    ax.set_xlabel("paired logit margin drop = margin(PPP) - margin(variant)   "
                  "(above 0: the N image moved the answer towards No)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=3, frameon=False, fontsize=9)
    set_title(ax, "Margin drop when one P image is replaced by an N image", subtitle)
    save(fig, out_path)


def plot_tendency_vs_margin_drop(table: dict[str, dict], subtitle: str, out_path: Path) -> None:
    """클래스마다 점 하나: 가로는 PPP의 평균 margin(성향), 세로는 margin 하락(세 변형 합침), 색은 N 1장일 때의 Yes 비율

    점선 위에서는 하락이 성향과 같아서 N 1장일 때의 평균 margin이 0이다. 점선 아래의 클래스는 그 평균이 0보다 크다.
    """
    names = list(table)
    tendency = [table[name]["mean_margin_PPP"] for name in names]
    drops = [table[name][f"margin_drop_{ONE_N}"] for name in names]
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    ax.axhline(0, color=GRID, linewidth=1)
    ax.axvline(0, color=GRID, linewidth=1)
    ax.axline((0, 0), slope=1, color=MUTED, linewidth=1, linestyle="--",
              label="margin drop = PPP margin (mean one-N margin is 0); below the line it stays above 0")
    points = ax.scatter(tendency, drops, c=[table[name][f"yes_rate_{ONE_N}"] for name in names], vmin=0, vmax=1,
                        cmap=LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP), s=80, edgecolors=INK_SECONDARY,
                        linewidths=0.6, zorder=3)
    for name, x, y in zip(names, tendency, drops):
        ax.annotate(name, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    colorbar = fig.colorbar(points, ax=ax, format=PercentFormatter(1.0), shrink=0.7)
    colorbar.set_label("share of 'Yes' answers in one-N cases (False-Yes)")
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0)
    ax.grid(True)
    ax.tick_params(length=0)
    ax.set_xlabel("class tendency: mean logit margin of the class's PPP cases")
    ax.set_ylabel("paired margin drop, mean of PPN, PNP, NPP")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), frameon=False, fontsize=9)
    set_title(ax, "Class tendency and margin drop", subtitle)
    save(fig, out_path)


def plot_within_class_auroc(table: dict[str, dict], ranges: dict[str, tuple[float, float]], pooled: float,
                            subtitle: str, out_path: Path) -> None:
    """클래스마다의 AUROC(PPP 대 N 1장)와, 클래스를 섞어 한 번에 계산한 AUROC

    ranges: 클래스마다, 이미지 한 장을 뺐을 때 그 클래스의 AUROC가 움직이는 (최솟값, 최댓값). 가는 선으로 그린다
    pooled: 모든 클래스의 케이스를 섞어 계산한 AUROC
    """
    names = list(table)
    summary_y = len(names) + 0.5
    aurocs = [table[name][f"auroc_{ONE_N}"] for name in names]
    fig, ax = plt.subplots(figsize=(8, 8.4))
    ax.axvspan(*NO_DIFFERENCE, color=GRID, linewidth=0,
               label=f"no practical difference ({NO_DIFFERENCE[0]} - {NO_DIFFERENCE[1]})")
    ax.axvline(0.5, color=MUTED, linewidth=1)
    ax.axvline(pooled, color=ORANGE, linewidth=1.5, linestyle="--", label=f"all classes pooled ({pooled:.3f})")
    for y, name in enumerate(names):
        ax.plot(ranges[name], [y, y], color=BLUE, linewidth=1.5, alpha=0.5)
    ax.scatter(aurocs, range(len(names)), s=40, color=BLUE, zorder=3,
               label="within one class (thin line: range when one image is left out)")
    draw_class_mean(ax, aurocs, summary_y, BLUE)
    ax.set_xlim(0, 1.02)
    style_class_axis(ax, names, summary_y)
    ax.set_xlabel("AUROC of the logit margin, PPP vs one-N cases   (0.5: no ranking, 1: PPP always higher)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), frameon=False, fontsize=9)
    set_title(ax, "Within-class AUROC, PPP vs cases with one N image", subtitle)
    save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="클래스 내부 민감도 분석의 그림")
    parser.add_argument("run", help="outputs/predictions/01_baseline/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    pairs_by_class = class_pairs(load_rows(PREDICTIONS_DIR / EXPERIMENT / args.run))
    table = {name: class_values(pairs) for name, pairs in pairs_by_class.items()}
    table = dict(sorted(table.items(), key=lambda item: -item[1]["mean_margin_PPP"]))  # PPP margin이 높은 클래스부터
    all_pairs = [pair for pairs in pairs_by_class.values() for pair in pairs]
    ranges = {name: leave_one_image_out(pairs_by_class[name])[f"auroc_{ONE_N}"] for name in table}
    subtitle = f"{args.run}, {len(table)} classes, {len(all_pairs)} pairs, classes sorted by PPP margin"

    plt.rcParams.update(STYLE)
    out_dir = FIGURES_DIR / EXPERIMENT / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_margin_drop(table, subtitle, out_dir / "class_margin_drop.png")
    plot_tendency_vs_margin_drop(table, subtitle, out_dir / "class_tendency_vs_margin_drop.png")
    plot_within_class_auroc(table, ranges, class_values(all_pairs)[f"auroc_{ONE_N}"], subtitle,
                            out_dir / "class_within_auroc.png")


if __name__ == "__main__":
    main()
