"""
hidden state 실행의 사진 비교 그림을 outputs/figures/03_hidden_states/<실행 이름>/에 저장
python experiments/03_hidden_states/plot.py qwen2.5-vl-3b_float32

그림 4장. 숫자는 summarize.py와 같은 함수로 계산한 pair 중앙값이다.
  1_slot_change.png            같은 사진을 다른 자리로 옮겼을 때
  2_earlier_image_changed.png  자리는 그대로이고 앞의 사진만 바꿨을 때
  3_position_control.png       위치를 통제한 pair와 나머지 pair
  4_token_vs_mean.png          토큰별 유사도와 평균끼리의 유사도
same_prefix 기준선이 큰 실행(계산 오차가 섞인 실행)에는 그림을 만들지 않는다.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mivlm.analysis.comparisons import (
    ONE_IMAGE_CHANGED, is_position_controlled, load_comparison_rows, median_by_layer, same_prefix_baseline,
)
from mivlm.analysis.hidden_states import SAME_WITHIN
from mivlm.analysis.pair_record import COMPARISONS
from mivlm.config import FIGURES_DIR, PREDICTIONS_DIR
from mivlm.plotting import AQUA, BLUE, MUTED, ORANGE, STYLE, figure_title, save, set_title, style_layer_axis

EXPERIMENT = "03_hidden_states"
Y_LABEL = "cosine similarity, token by token"


def comparison_name(index: int) -> str:
    """COMPARISONS의 순번(0부터)을 그림에 쓸 이름으로 바꾼다. 예: "N1: PPN slot 3 vs PNN slot 3" """
    _, image, (a_variant, a_slot), (b_variant, b_slot) = COMPARISONS[index]
    return f"{image}: {a_variant} slot {a_slot} vs {b_variant} slot {b_slot}"


def draw_median(ax, rows: list[dict], metric: str, color: str, label: str, band: bool = True, **line_style) -> None:
    """rows의 metric을 층별로 그린다: 선은 pair 중앙값, 띠는 가운데 50%의 pair가 들어가는 범위"""
    values = torch.stack([row[metric] for row in rows])
    layers = range(values.shape[1])
    if band:
        low, high = values.quantile(torch.tensor([0.25, 0.75]), dim=0)
        ax.fill_between(layers, low, high, color=color, alpha=0.15, linewidth=0)
    ax.plot(layers, median_by_layer(rows, metric), color=color, linewidth=2, label=label, **line_style)


def plot_slot_change(by_index: dict, subtitle: str, out_path: Path) -> None:
    """표 1의 열 1~3과 12: 같은 N1을 다른 자리에 놓았을 때 그 사진의 토큰 벡터가 얼마나 달라지는가"""
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    draw_median(ax, by_index[0], "token_cosine", BLUE, "slot 1 vs slot 2")
    draw_median(ax, by_index[1], "token_cosine", ORANGE, "slot 1 vs slot 3")
    draw_median(ax, by_index[2], "token_cosine", AQUA, "slot 2 vs slot 3")
    draw_median(ax, by_index[11], "token_cosine", MUTED, "nothing changed (reference)", band=False, linestyle="--")
    style_layer_axis(ax, y_min=0.86, y_label=Y_LABEL)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    set_title(ax, "The same image, moved to another slot", subtitle)
    save(fig, out_path)


def plot_earlier_image_changed(by_index: dict, subtitle: str, out_path: Path) -> None:
    """표 1의 열 4~6, 9~11: 사진과 자리는 그대로이고 앞의 사진을 바꿨을 때. 어느 자리를 바꿨는지로 색을 나눈다"""
    panels = {  # 칸 제목 -> (자리 1을 바꾼 비교, 자리 2를 바꾼 비교, 둘 다 바꾼 비교)의 순번
        "P3 in slot 3": (9, 8, 10),
        "N1 in slot 3 (earlier images changed from P to N)": (4, 3, 5),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (name, indexes) in zip(axes, panels.items()):
        for index, color, label in zip(indexes, [BLUE, ORANGE, AQUA],
                                       ["slot 1 image changed", "slot 2 image changed", "both changed"]):
            draw_median(ax, by_index[index], "token_cosine", color, label)
        style_layer_axis(ax, y_min=0.92, y_label=Y_LABEL if ax is axes[0] else None, note=ax is axes[1])
        ax.text(0.5, 1.01, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    figure_title(fig, "Same image, same slot, an earlier image changed", subtitle)
    save(fig, out_path)


def plot_position_control(by_index: dict, subtitle: str, out_path: Path) -> None:
    """표 4: 토큰 인덱스와 position id가 모두 같은 pair만 골라도 차이가 남는가"""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True, sharey=True)
    for ax, index in zip(axes.flat, ONE_IMAGE_CHANGED):
        controlled = [row for row in by_index[index] if is_position_controlled(row)]
        others = [row for row in by_index[index] if not is_position_controlled(row)]
        draw_median(ax, others, "token_cosine", ORANGE, "other pairs", band=False)
        draw_median(ax, controlled, "token_cosine", BLUE, "same token index and position ids", band=False,
                    linestyle="--")
        style_layer_axis(ax, y_min=0.94, y_label=Y_LABEL if ax in axes[:, 0] else None, note=ax is axes[1, 2])
        if ax in axes[0]:
            ax.set_xlabel("")
        ax.text(0.5, 1.01, f"{comparison_name(index)}\n{len(controlled)} controlled pairs, {len(others)} others",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
    axes[1, 0].legend(loc="lower left", frameon=False, fontsize=9)
    figure_title(fig, "Pairs with the position controlled, and the other pairs", subtitle)
    save(fig, out_path)


def plot_token_vs_mean(by_index: dict, subtitle: str, out_path: Path) -> None:
    """표 1과 표 3: 토큰끼리 비교한 값과, 토큰을 먼저 평균 낸 벡터끼리 비교한 값"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, index in zip(axes, [1, 8]):
        draw_median(ax, by_index[index], "token_cosine", BLUE, "token by token, then averaged")
        draw_median(ax, by_index[index], "cosine_of_means", ORANGE, "tokens averaged first, then compared")
        style_layer_axis(ax, y_min=0.86, y_label="cosine similarity" if ax is axes[0] else None, note=ax is axes[1])
        ax.text(0.5, 1.01, comparison_name(index), transform=ax.transAxes, ha="center", va="bottom", fontsize=10,
                fontweight="bold")
    axes[0].legend(loc="lower left", frameon=False, fontsize=9)
    figure_title(fig, "Token-by-token similarity, and similarity of mean-pooled vectors", subtitle)
    save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="hidden state 실행의 사진 비교 그림")
    parser.add_argument("run", help="outputs/predictions/03_hidden_states/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    rows = load_comparison_rows(PREDICTIONS_DIR / EXPERIMENT / args.run)
    baseline = same_prefix_baseline(rows)
    if baseline >= SAME_WITHIN:
        raise SystemExit(f"same_prefix 기준선이 {baseline:.3g}로 기준 {SAME_WITHIN}보다 크다. "
                         f"이 실행의 사진 비교에는 계산 오차가 섞여 있어서 그림을 만들지 않는다")
    by_index = {index: [row for row in rows if row["index"] == index] for index in range(len(COMPARISONS))}
    medians = f"{args.run}, median over {len(by_index[0])} pairs"
    subtitle = f"{medians}. Band = middle 50% of pairs"

    plt.rcParams.update(STYLE)
    out_dir = FIGURES_DIR / EXPERIMENT / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_slot_change(by_index, subtitle, out_dir / "1_slot_change.png")
    plot_earlier_image_changed(by_index, subtitle, out_dir / "2_earlier_image_changed.png")
    plot_position_control(by_index, medians, out_dir / "3_position_control.png")  # 이 그림에는 띠를 그리지 않는다
    plot_token_vs_mean(by_index, subtitle, out_dir / "4_token_vs_mean.png")


if __name__ == "__main__":
    main()
