"""
linear probe 결과 그림을 outputs/figures/04_probe/<실행 이름>/에 저장
python experiments/04_probe/plot.py qwen2.5-vl-3b

그림 3장. 값은 summarize.py의 표와 같다 (학습에 쓰지 않은 클래스의 케이스에서, 층마다).
  1_presence.png  N이 있는가의 정확도: 전체, 그리고 케이스 종류별
  2_slot.png      N이 어느 자리인가의 정확도: 전체, 모델의 답별, 자리별
  3_auroc.png     N이 있는가의 AUROC: 전체, 그리고 모델이 Yes라고 답한 케이스만. 모델의 logit margin만 쓴 기준선과 같이
모델의 답은 logit margin의 부호로 본다 (양수면 Yes).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mivlm.analysis.probe import accuracy_by_layer
from mivlm.config import FIGURES_DIR, PREDICTIONS_DIR
from mivlm.evaluation.auroc import auroc
from mivlm.plotting import AQUA, BLUE, MUTED, ORANGE, STYLE, figure_title, save, style_layer_axis

EXPERIMENT = "04_probe"
Y_LABEL = "probe accuracy on held-out classes"


def draw_panel(ax, name: str, lines: dict, references: dict, y_label: str | None, note: bool) -> None:
    """칸 하나를 그린다

    lines: {이름: (층마다의 값, 색, 선 모양)}
    references: {이름: (값, 선 모양)}. 층과 무관한 기준값을 가는 가로선으로 그리고, 범례의 이름 뒤에 값을 적는다
    범례는 칸 아래에 둔다. 선이 0부터 1까지 어디든 지나가서 칸 안에는 늘 비어 있는 자리가 없다
    """
    for label, (values, color, linestyle) in lines.items():
        ax.plot(range(len(values)), values, color=color, linewidth=2, linestyle=linestyle, label=label)
    for label, (value, linestyle) in references.items():
        ax.axhline(value, color=MUTED, linewidth=1, linestyle=linestyle, label=f"{label} ({value:.3f})")
    style_layer_axis(ax, y_min=0, y_label=y_label, note=note)
    ax.text(0.5, 1.01, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), frameon=False, fontsize=9, handlelength=2.6)


def plot_presence(task: dict, subtitle: str, out_path: Path) -> None:
    """과제 "N이 있는가" (레이블 0 = PPP, 1 = N이 하나)"""
    labels, predictions = task["labels"], task["scores"].argmax(dim=-1)
    everything = torch.ones_like(labels, dtype=torch.bool)
    one_n = labels == 1
    model_yes = task["logit_margins"] > 0
    model_accuracy = torch.where(one_n, ~model_yes, model_yes).float().mean().item()  # PPP는 Yes가, 나머지는 No가 정답

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    draw_panel(axes[0], "All cases", {
        "probe": (accuracy_by_layer(predictions, labels, everything), BLUE, "-"),
        "probe trained on shuffled labels": (accuracy_by_layer(task["control_scores"].argmax(dim=-1), labels,
                                                               everything), MUTED, "--"),
    }, {
        "always answering 'an N is present'": (one_n.float().mean().item(), ":"),
        "the model's own answers": (model_accuracy, "-."),
    }, y_label=Y_LABEL, note=False)
    draw_panel(axes[1], "By kind of case", {
        "PPP (no N)": (accuracy_by_layer(predictions, labels, ~one_n), BLUE, "-"),
        "one N, model answered No (correct)": (accuracy_by_layer(predictions, labels, one_n & ~model_yes), ORANGE, "-"),
        "one N, model answered Yes (False-Yes)": (accuracy_by_layer(predictions, labels, one_n & model_yes), AQUA, "-"),
    }, {}, y_label=None, note=True)
    figure_title(fig, "Linear probe at the answer position: is an N image present?", subtitle)
    save(fig, out_path)


def plot_slot(task: dict, subtitle: str, out_path: Path) -> None:
    """과제 "N이 어느 자리인가" (레이블 0, 1, 2 = 자리 1, 2, 3). 케이스는 모두 N이 하나인 변형"""
    labels, predictions = task["labels"], task["scores"].argmax(dim=-1)
    everything = torch.ones_like(labels, dtype=torch.bool)
    model_yes = task["logit_margins"] > 0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    draw_panel(axes[0], "All cases", {
        "probe": (accuracy_by_layer(predictions, labels, everything), BLUE, "-"),
        "probe trained on shuffled labels": (accuracy_by_layer(task["control_scores"].argmax(dim=-1), labels,
                                                               everything), MUTED, "--"),
    }, {"chance": (1 / 3, ":")}, y_label=Y_LABEL, note=False)
    draw_panel(axes[1], "By the model's answer", {
        "model answered No (correct)": (accuracy_by_layer(predictions, labels, ~model_yes), ORANGE, "-"),
        "model answered Yes (False-Yes)": (accuracy_by_layer(predictions, labels, model_yes), AQUA, "-"),
    }, {}, y_label=None, note=False)
    draw_panel(axes[2], "By the slot of the N image", {
        f"slot {slot + 1} ({variant})": (accuracy_by_layer(predictions, labels, labels == slot), color, "-")
        for slot, (variant, color) in enumerate(zip(["NPP", "PNP", "PPN"], [BLUE, ORANGE, AQUA]))
    }, {}, y_label=None, note=True)
    figure_title(fig, "Linear probe at the answer position: which slot holds the N image?", subtitle)
    save(fig, out_path)


def plot_auroc(task: dict, subtitle: str, out_path: Path) -> None:
    """과제 "N이 있는가"의 AUROC. probe의 점수와, probe 없이 모델의 logit margin만 쓴 기준선을 같이 그린다"""
    labels, model_yes = task["labels"], task["logit_margins"] > 0
    n_score = task["scores"][..., 1] - task["scores"][..., 0]  # "N 있음" 쪽 점수, shape (층 수 + 1, 케이스 수)
    margin_score = -task["logit_margins"][None]  # margin이 낮을수록 "N 있음" 쪽. 층이 하나인 것처럼 다룬다

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    panels = {
        "All cases (PPP vs one N)": torch.ones_like(labels, dtype=torch.bool),
        "Cases the model answered Yes (PPP vs False-Yes)": model_yes,
    }
    for ax, (name, mask) in zip(axes, panels.items()):
        draw_panel(ax, name, {"probe": (auroc(n_score[:, mask], labels[mask]), BLUE, "-")}, {
            "the model's own Yes/No margin, no probe": (auroc(margin_score[:, mask], labels[mask]).item(),
                                                        "-."),
            "no separation": (0.5, ":"),
        }, y_label="AUROC on held-out classes" if ax is axes[0] else None, note=ax is axes[1])
    figure_title(fig, "Linear probe at the answer position: AUROC for 'is an N image present?'", subtitle)
    save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="linear probe 결과 그림")
    parser.add_argument("run", help="outputs/predictions/04_probe/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    saved = torch.load(PREDICTIONS_DIR / EXPERIMENT / args.run / "probe.pt", map_location="cpu", weights_only=True)
    tasks = saved["tasks"]

    plt.rcParams.update(STYLE)
    out_dir = FIGURES_DIR / EXPERIMENT / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    setting = f"class-level {saved['settings']['n_folds']}-fold cross-validation. N = image without the asked object"
    presence_subtitle = f"{args.run}, {len(tasks['presence']['labels'])} cases, {setting}"
    plot_presence(tasks["presence"], presence_subtitle, out_dir / "1_presence.png")
    plot_slot(tasks["slot"], f"{args.run}, {len(tasks['slot']['labels'])} cases, {setting}", out_dir / "2_slot.png")
    plot_auroc(tasks["presence"], presence_subtitle, out_dir / "3_auroc.png")


if __name__ == "__main__":
    main()
