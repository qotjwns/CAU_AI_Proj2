"""
baseline 결과 요약: 변형별, 모드별, 클래스별 지표를 표로 출력. 실행 이름을 두 개 주면 둘을 비교한다
python experiments/01_baseline/summarize.py qwen2.5-vl-3b
python experiments/01_baseline/summarize.py qwen2.5-vl-3b qwen2.5-vl-3b_float32
"""
import argparse

from mivlm.config import PREDICTIONS_DIR
from mivlm.evaluation.metrics import BOUNDARY_MARGIN, compare_runs, load_rows, summarize, summarize_by

EXPERIMENT = "01_baseline"
VARIANTS = ["PPP", "PPN", "PNP", "NPP", "PNN", "NNN"]  # N 개수 순서 (0, 1, 1, 1, 2, 3)
COLUMNS = ["n", "accuracy", "false_yes_rate", "false_no_rate", "yes_rate", "mean_margin", "n_tie", "n_boundary", "n_none"]


def format_value(column: str, value) -> str:
    if value is None:
        return "-"
    if column in ("accuracy", "false_yes_rate", "false_no_rate", "yes_rate"):
        return f"{value:.1%}"
    if column == "mean_margin":
        return f"{value:+.3f}"
    return str(value)


def print_table(title: str, summaries: dict[str, dict]) -> None:
    """summaries: {행 이름: summarize 결과}"""
    print(f"\n[{title}]")
    print(f"{'':<14}" + "".join(f"{column:>16}" for column in COLUMNS))
    for name, summary in summaries.items():
        print(f"{name:<14}" + "".join(f"{format_value(column, summary[column]):>16}" for column in COLUMNS))


def print_run(run: str, rows: list[dict]) -> None:
    print(f"\n{'=' * 20} {run} {'=' * 20}")
    print_table("전체", {"all": summarize(rows)})

    by_variant = summarize_by(rows, "variant")
    print_table("변형별", {variant: by_variant[variant] for variant in VARIANTS if variant in by_variant})

    print_table("모드별", {mode: summarize([row for row in rows if mode in row["modes"]]) for mode in "AB"})

    print_table("클래스별", summarize_by(rows, "object"))

    # 클래스 x 변형의 yes_rate. plot.py 히트맵과 같은 값이다
    print("\n[클래스 x 변형: Yes로 답한 비율]")
    print(f"{'':<14}" + "".join(f"{variant:>8}" for variant in VARIANTS))
    for name in summarize_by(rows, "object"):
        by_variant_in_class = summarize_by([row for row in rows if row["object"] == name], "variant")
        cells = [by_variant_in_class[variant]["yes_rate"] if variant in by_variant_in_class else None for variant in VARIANTS]
        print(f"{name:<14}" + "".join(f"{format_value('yes_rate', cell):>8}" for cell in cells))

    away_from_boundary = summarize_by([row for row in rows if abs(row["logit_margin"]) > BOUNDARY_MARGIN], "variant")
    print_table(f"변형별, 경계 케이스(|margin| <= {BOUNDARY_MARGIN}) 제외",
                {variant: away_from_boundary[variant] for variant in VARIANTS if variant in away_from_boundary})


def main() -> None:
    parser = argparse.ArgumentParser(description="baseline 결과 요약")
    parser.add_argument("run", help="outputs/predictions/01_baseline/ 아래의 실행 폴더 이름")
    parser.add_argument("compare_with", nargs="?", help="비교할 두 번째 실행 폴더 이름")
    args = parser.parse_args()

    rows = load_rows(PREDICTIONS_DIR / EXPERIMENT / args.run)
    print_run(args.run, rows)

    if args.compare_with:
        other_rows = load_rows(PREDICTIONS_DIR / EXPERIMENT / args.compare_with)
        print_run(args.compare_with, other_rows)
        print(f"\n{'=' * 20} {args.run} 대 {args.compare_with} {'=' * 20}")
        for key, value in compare_runs(rows, other_rows).items():
            print(f"{key:<22}{value:+.4f}" if isinstance(value, float) else f"{key:<22}{value}")


if __name__ == "__main__":
    main()
