"""
결정 기준 이동 분석의 그림을 outputs/figures/01_baseline/<실행 이름>/에 저장
python experiments/01_baseline/plot_logit_bias.py qwen2.5-vl-3b_float32

그림 1장: logit_bias_tradeoff.png
    가로는 원래 맞힌 PPP 중 No로 바뀐 비율(손상), 세로는 원래 False-Yes였던 케이스 중 No로 바뀐 비율(교정)이다.
    선은 모든 케이스에 같은 기준 b를 적용하고 b를 0부터 올려 가며 그린 곡선이다 (같은 데이터로 그린 기술 통계).
    점은 기준을 평가하는 쪽 밖에서 고른 결과다: 손상 한도의 기준선(2%, 5%, 10%)과 global, 그리고 class_cv와 class_oracle.
    값은 logit_bias.py의 표와 같은 함수로 계산하고, 케이스를 모두 합쳐 센 비율이다.
이후의 개입은 이 그림 위에 같은 두 축으로 올려서 비교한다. bf16 실행은 margin이 0.125 간격이라 곡선의 점이 듬성하다.
"""
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from mivlm.analysis.logit_bias import (
    DAMAGE_LIMITS, bias_metrics, damage_limited_biases, no_shift_candidates, selection_rules, tradeoff_curve,
)
from mivlm.config import FIGURES_DIR, PREDICTIONS_DIR
from mivlm.evaluation.metrics import ONE_N_VARIANTS, load_rows
from mivlm.plotting import AQUA, BLUE, INK_SECONDARY, MUTED, ORANGE, STYLE, SURFACE, VIOLET, save, set_title

EXPERIMENT = "01_baseline"


def main() -> None:
    parser = argparse.ArgumentParser(description="결정 기준 이동 분석의 그림")
    parser.add_argument("run", help="outputs/predictions/01_baseline/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    rows = [row for row in load_rows(PREDICTIONS_DIR / EXPERIMENT / args.run)
            if row["variant"] in ["PPP", *ONE_N_VARIANTS]]
    one_class = {row["object"]: row["object"] for row in rows}
    curve = tradeoff_curve(rows)
    rules = selection_rules(rows, one_class, no_shift_candidates)
    # 점 이름 -> (행마다의 기준, 색, 표시 모양)
    points = {
        **{f"damage limit {limit:.0%}, threshold from other classes":
           (damage_limited_biases(rows, one_class, limit), ORANGE, marker)
           for limit, marker in zip(DAMAGE_LIMITS, ["v", "s", "^"])},
        "global: best balanced accuracy on other classes": (rules["global"], BLUE, "o"),
        "class_cv: per class, from the class's other groups": (rules["class_cv"], AQUA, "D"),
        "class_oracle: per class, upper bound": (rules["class_oracle"], VIOLET, "X"),
    }

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.axline((0, 0), slope=1, color=MUTED, linewidth=1, linestyle="--", label="equal shares (no gain)")
    ax.plot([point["ppp_damage_rate"] for point in curve], [point["false_yes_correction_rate"] for point in curve],
            color=INK_SECONDARY, linewidth=1.5, marker="o", markersize=2.5,
            label="one threshold b for all cases, b raised from 0 (same data)")
    for label, (biases, color, marker) in points.items():
        pooled = bias_metrics(rows, biases)
        ax.scatter(pooled["ppp_damage_rate"], pooled["false_yes_correction_rate"], s=90, color=color, marker=marker,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True)
    ax.tick_params(length=0)
    ax.set_xlabel("PPP damage: share of originally correct PPP answers turned to No")
    ax.set_ylabel("False-Yes correction: share of original False-Yes answers turned to No")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), frameon=False, fontsize=9)
    n_ppp, n_false_yes = curve[0]["n_ppp_yes"], curve[0]["n_false_yes"]
    set_title(ax, "Moving the Yes threshold: False-Yes corrected against PPP damaged",
              f"{args.run}, {n_false_yes} original False-Yes cases, {n_ppp} originally correct PPP cases")
    out_dir = FIGURES_DIR / EXPERIMENT / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    save(fig, out_dir / "logit_bias_tradeoff.png")


if __name__ == "__main__":
    main()
