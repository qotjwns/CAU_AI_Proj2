"""
분해 실험 그림을 outputs/figures/02_decomposition/<실행 이름>/에 저장 (전체 실행 결과용)
python experiments/02_decomposition/plot.py qwen2.5-vl-3b

그림 5장. 파일 이름의 번호는 summarize.py가 출력하는 표의 번호이고, 숫자도 같은 함수로 계산한다.
  1_main_metrics.png           전체 정확도, N 1장 False-Yes, PPP 정확도
  2_yes_rate_by_variant.png    변형별 Yes로 답한 비율
  4_a_false_yes_breakdown.png  A가 False-Yes인 케이스를 B, C는 맞혔는가
  5_image_errors.png           이미지별 판단의 오류율
  6_hidden_image_errors.png    AND가 가리는 오류
"""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from mivlm.config import PREDICTIONS_DIR, PROJECT_ROOT
from mivlm.data.cases import load_cases
from mivlm.evaluation.decomposition import (
    combine_per_image, combine_single_image, images_all_correct, load_per_image_rows, load_single_image_rows,
)
from mivlm.evaluation.metrics import VARIANT_ORDER, load_rows, summarize, summarize_by
from mivlm.plotting import AQUA, BLUE, INK, INK_SECONDARY, MUTED, ORANGE, STYLE, SURFACE, VIOLET, save, set_title

EXPERIMENT = "02_decomposition"
ONE_N_VARIANTS = ["PPN", "PNP", "NPP"]  # N이 1장인 변형

# 실험마다 색을 하나 정해 모든 그림에서 같은 색을 쓴다
COLORS = {"A": BLUE, "B": ORANGE, "C": AQUA}
NAMES = {
    "A": "A: 3 images, asked about all 3 at once",
    "B": "B: 1 image at a time, 3 answers combined with AND",
    "C": "C: 3 images, asked slot by slot, 3 answers combined with AND",
}


def draw_segments(ax, y: float, counts: list[int], styles: list[tuple[str, str]], total: int) -> None:
    """가로 막대 하나를 구간으로 나눠 그린다. 가로축은 total에서 차지하는 비율(0~1)이다

    counts: 구간에 속한 개수를 왼쪽부터. styles: 구간마다 (채우는 색, 글자색)
    구간 사이는 배경색 선으로 띄우고, 글자가 들어갈 만큼 넓은 구간에만 비율을 적는다.
    """
    left = 0.0
    for count, (color, text_color) in zip(counts, styles):
        if count == 0:
            continue
        share = count / total
        ax.barh(y, share, left=left, height=0.5, color=color, edgecolor=SURFACE, linewidth=2)
        if share >= 0.07:
            ax.text(left + 0.012, y, f"{share:.1%}", va="center", color=text_color, fontsize=9, fontweight="bold")
        left += share


def plot_main_metrics(case_rows: dict[str, list[dict]], run: str, out_path: Path) -> None:
    """표 1: 전체 정확도, N이 1장인 변형의 False-Yes, PPP 정확도를 A, B, C끼리 비교한다"""
    # 그릴 숫자: 지표마다 A, B, C 순서의 값 3개
    accuracy, one_n_false_yes, ppp_accuracy = [], [], []
    for rows in case_rows.values():
        one_n_rows = [row for row in rows if row["variant"] in ONE_N_VARIANTS]
        ppp_rows = [row for row in rows if row["variant"] == "PPP"]
        accuracy.append(summarize(rows)["accuracy"])
        one_n_false_yes.append(summarize(one_n_rows)["false_yes_rate"])
        ppp_accuracy.append(summarize(ppp_rows)["accuracy"])
    panels = {
        "Accuracy, all cases\n(higher is better)": accuracy,
        "False-Yes, variants with one N\n(lower is better)": one_n_false_yes,
        "Accuracy, PPP\n(higher is better)": ppp_accuracy,
    }

    # 그리기: 지표마다 칸 하나. 높을수록 좋은 지표와 낮을수록 좋은 지표를 한 칸에 섞지 않는다
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), sharey=True)
    colors = [COLORS[name] for name in case_rows]
    for ax, (title, values) in zip(axes, panels.items()):
        bars = ax.bar(list(case_rows), values, width=0.3, color=colors)
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
        ax.set_title(title, loc="left", fontsize=10, fontweight="normal", color=INK_SECONDARY)
        ax.set_ylim(0, 1.1)  # 100%에 가까운 막대 위에도 숫자가 들어갈 자리
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(axis="y")
        ax.tick_params(length=0)
    handles = [Patch(color=COLORS[name], label=NAMES[name]) for name in case_rows]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.02), frameon=False)
    fig.suptitle(f"Main results: {run}", x=0.07, y=1.08, ha="left", fontsize=13, fontweight="bold")
    save(fig, out_path)


def plot_yes_rate_by_variant(case_rows: dict[str, list[dict]], run: str, out_path: Path) -> None:
    """표 2: 변형마다 Yes로 답한 비율. PPP만 정답이 Yes이고, 나머지에서는 이 비율이 False-Yes rate다"""
    by_variant = {name: summarize_by(rows, "variant") for name, rows in case_rows.items()}
    step = 0.3  # 같은 변형 안에서 이웃한 막대 사이의 거리 (변형 사이의 거리는 1)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, name in enumerate(case_rows):
        rates = [by_variant[name][variant]["yes_rate"] for variant in VARIANT_ORDER]
        # 변형의 가운데 자리(0, 1, 2, ...)에서 A는 왼쪽, B는 가운데, C는 오른쪽
        positions = [center + (i - 1) * step for center in range(len(VARIANT_ORDER))]
        bars = ax.bar(positions, rates, width=0.2, color=COLORS[name], label=NAMES[name])
        ax.bar_label(bars, labels=[f"{rate:.1%}" for rate in rates], padding=3, fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(range(len(VARIANT_ORDER)), ["PPP\n(Yes is correct)"] + VARIANT_ORDER[1:])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(axis="y")
    ax.tick_params(length=0)
    ax.set_xlabel("images in slots 1-3 (P: object present, N: object absent)")
    ax.set_ylabel("share of 'Yes' answers")
    ax.legend(frameon=False, loc="upper right")
    set_title(ax, "'Yes' answers by variant", f"{run}, {by_variant['A']['PPP']['n']} cases per variant")
    save(fig, out_path)


def plot_a_false_yes_breakdown(case_rows: dict[str, list[dict]], run: str, out_path: Path) -> None:
    """표 4: A가 False-Yes인 케이스에서 B와 C(AND로 합친 답)는 무엇이라 답했는가"""
    # 그릴 숫자: (B의 답, C의 답)마다 케이스 수
    b_answers = {row["case_id"]: row["prediction"] for row in case_rows["B"]}
    c_answers = {row["case_id"]: row["prediction"] for row in case_rows["C"]}
    a_false_yes = [row for row in case_rows["A"] if row["label"] == "No" and row["prediction"] == "Yes"]
    pair_counts = Counter((b_answers[row["case_id"]], c_answers[row["case_id"]]) for row in a_false_yes)
    # 구간: (B의 답, C의 답), 이름, (채우는 색, 글자색)
    parts = [
        (("No", "No"), "B and C both answer No (correct)", (MUTED, INK)),
        (("No", "Yes"), "only B answers No", (COLORS["B"], "white")),
        (("Yes", "No"), "only C answers No", (COLORS["C"], "white")),
        (("Yes", "Yes"), "B and C also answer Yes", (INK, "white")),
    ]
    counts = [pair_counts[answers] for answers, _, _ in parts]
    styles = [style for _, _, style in parts]

    # 그리기: 막대 하나를 네 구간으로. 구간별 개수는 범례에 적는다
    fig, ax = plt.subplots(figsize=(9, 0.55))
    draw_segments(ax, 0, counts, styles, len(a_false_yes))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 0.4)  # 막대(높이 0.5)와 부제목 사이를 조금 띄운다
    ax.axis("off")
    handles = [Patch(color=color, label=f"{name}: {count:,}") for (_, name, (color, _)), count in zip(parts, counts)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 0), ncols=2, frameon=False)
    set_title(ax, "A's False-Yes cases: what do B and C answer?",
              f"{run}, {len(a_false_yes):,} cases where A answers Yes but the correct answer is No")
    save(fig, out_path)


def plot_image_errors(b_image_rows: list[dict], c_image_rows: list[dict], run: str, out_path: Path) -> None:
    """표 5: 이미지 하나에 대한 답이 틀린 비율. B는 그 이미지만 넣었을 때, C는 3장 중 한 자리를 지목했을 때"""
    # 그릴 숫자: B 전체와 C의 자리별 요약
    summaries = {"B": summarize(b_image_rows)}
    for slot in (1, 2, 3):
        summaries[f"C, slot {slot}"] = summarize([row for row in c_image_rows if row["slot"] == slot])
    colors = [COLORS["B"], COLORS["C"], COLORS["C"], COLORS["C"]]
    panels = {
        "P image answered No (object missed)": "false_no_rate",
        "N image answered Yes (object made up)": "false_yes_rate",
    }

    # 그리기: 오류 종류마다 칸 하나. 두 칸이 같은 세로축을 쓴다
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (title, column) in zip(axes, panels.items()):
        rates = [summary[column] for summary in summaries.values()]
        bars = ax.bar(list(summaries), rates, width=0.3, color=colors)
        ax.bar_label(bars, labels=[f"{rate:.1%}" for rate in rates], padding=3)
        ax.set_title(title, loc="left", fontsize=10, fontweight="normal", color=INK_SECONDARY)
        ax.margins(y=0.15)  # 막대 위에 숫자가 들어갈 자리
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.grid(axis="y")
        ax.tick_params(length=0)
    handles = [Patch(color=COLORS[name], label=NAMES[name]) for name in ("B", "C")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.02), frameon=False)
    fig.suptitle(f"Wrong answers about a single image: {run}", x=0.07, y=1.04, ha="left",
                 fontsize=13, fontweight="bold")
    save(fig, out_path)


def plot_hidden_image_errors(case_rows: dict[str, list[dict]], run: str, out_path: Path) -> None:
    """표 6: N이 1장인 케이스(정답 No)에서 AND로 합친 답이 가리는 이미지별 오류

    AND는 N 자리만 No라고 하면 No가 되므로, P 자리를 놓친 답도 케이스 단위에서는 정답으로 세어진다.
    판정 불가인 답이 있으면 막대가 100%에 못 미친다.
    """
    names = [
        "AND answers No, all 3 image answers correct",
        "AND answers No, but a P image is not answered Yes",
        "AND answers Yes (False-Yes)",
    ]
    styles = [(MUTED, INK), (VIOLET, "white"), (INK, "white")]  # 구간마다 (채우는 색, 글자색)

    fig, ax = plt.subplots(figsize=(9, 1.4))
    for y, name in enumerate(["B", "C"]):
        one_n_rows = [row for row in case_rows[name] if row["variant"] in ONE_N_VARIANTS]
        answered_no = [row for row in one_n_rows if row["prediction"] == "No"]
        n_all_correct = sum(images_all_correct(row) for row in answered_no)
        n_false_yes = sum(row["prediction"] == "Yes" for row in one_n_rows)
        draw_segments(ax, y, [n_all_correct, len(answered_no) - n_all_correct, n_false_yes], styles, len(one_n_rows))
    ax.set_yticks([0, 1], ["B", "C"])
    ax.invert_yaxis()  # B를 위에
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    handles = [Patch(color=color, label=name) for name, (color, _) in zip(names, styles)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.2), frameon=False)
    set_title(ax, "Image-level errors hidden by AND",
              f"{run}, {len(one_n_rows):,} cases with one N image (PPN, PNP, NPP), correct answer is No")
    save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="분해 실험 A, B, C 그림")
    parser.add_argument("run", help="실행 폴더 이름. 예: qwen2.5-vl-3b")
    args = parser.parse_args()

    # 이미지별 답 (B: 320개, C: 7,200개)
    b_image_rows = load_single_image_rows(PREDICTIONS_DIR / EXPERIMENT / "b_single_image" / args.run)
    c_image_rows = load_per_image_rows(PREDICTIONS_DIR / EXPERIMENT / "c_per_image_answers" / args.run)
    # 케이스 단위의 답 (각 2,400개). B와 C는 이미지별 답을 AND로 합친 것
    cases = load_cases()
    case_rows = {
        "A": load_rows(PREDICTIONS_DIR / "01_baseline" / args.run),
        "B": combine_single_image(cases, b_image_rows),
        "C": combine_per_image(cases, c_image_rows),
    }

    plt.rcParams.update(STYLE)
    out_dir = PROJECT_ROOT / "outputs" / "figures" / EXPERIMENT / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_main_metrics(case_rows, args.run, out_dir / "1_main_metrics.png")
    plot_yes_rate_by_variant(case_rows, args.run, out_dir / "2_yes_rate_by_variant.png")
    plot_a_false_yes_breakdown(case_rows, args.run, out_dir / "4_a_false_yes_breakdown.png")
    plot_image_errors(b_image_rows, c_image_rows, args.run, out_dir / "5_image_errors.png")
    plot_hidden_image_errors(case_rows, args.run, out_dir / "6_hidden_image_errors.png")


if __name__ == "__main__":
    main()
