"""
결정 기준 이동 분석 (fixed logit bias): 답을 읽는 기준만 옮기면 False-Yes가 얼마나 줄고 PPP가 얼마나 손상되는가
python experiments/01_baseline/logit_bias.py qwen2.5-vl-3b            # 주 분석
python experiments/01_baseline/logit_bias.py qwen2.5-vl-7b            # 모델 견고성 분석
python experiments/01_baseline/logit_bias.py qwen2.5-vl-3b_float32    # 정밀도 민감도 분석

저장된 A(01_baseline) 결과를 읽기만 한다. 모델은 쓰지 않는다. 표는 터미널에 출력하고,
클래스별 값은 outputs/tables/01_baseline/<실행 이름>_logit_bias.csv에도 저장한다. 그림은 plot_logit_bias.py가 그린다.
규칙과 지표의 정의는 mivlm.analysis.logit_bias 참고. 이후의 어떤 개입이든 이 기준선과 같은 PPP 손상 조건에서 비교한다.

규칙: margin > b이면 Yes. b = 0이 모델의 원래 답이다. 주 분석은 No 쪽으로의 이동(b >= 0)만 보고,
    b가 음수도 될 수 있는 양방향 보정은 맨 끝에 별도 분석으로 낸다.
기준 고르기: 균형 정확도(PPP 정확도와 N 1장 케이스의 정답률의 평균)가 가장 높은 b. 같으면 0에 가까운 것
    global = 다른 클래스에서 고름 / class_cv = 같은 클래스의 다른 그룹에서 고름 / class_oracle = 같은 클래스 전체에서 고름
손상 한도의 기준선: PPP 손상률이 2%, 5%, 10% 이하인 가장 큰 b를 다른 클래스에서 고른다. 평가한 클래스의 실제 손상률과
    한도를 넘었는지를 그대로 적는다.

판정 기준 (결과를 보기 전에 정했다. 결과를 본 뒤에 바꾸지 않는다). 둘 다 클래스마다의 균형 정확도 차이의 평균으로 본다
    G. global이 원래 답보다 낫다: 균형 정확도(global) - 균형 정확도(original)
    C. class_cv가 global보다 낫다: 균형 정확도(class_cv) - 균형 정확도(global)
    통계적 방향: 95% 구간 전체가 0보다 크면 나은 쪽, 전체가 0보다 작으면 나쁜 쪽, 그 밖에는 방향 불명
    실질적 효과: 구간 전체가 +2%p를 넘으면 실질적으로 나음, 전체가 -2%p 아래면 실질적으로 나쁨,
        전체가 -2%p ~ +2%p 안이면 실질적 차이 없음, 그 밖에는 미결정. 2%p는 연구자가 정한 기준이다
    두 판정은 따로 적는다. class_oracle과 맞바꿈 곡선에는 판정을 붙이지 않는다.
구간은 클래스를 단위로 10,000번 다시 뽑아 만든다 (seed 고정).
"""
import argparse
import csv
import math

from mivlm.analysis.class_sensitivity import (
    RUN_ROLES, class_pairs, direction_verdict, mean_interval, shared_images, without_images,
)
from mivlm.analysis.evidence_link import practical_verdict
from mivlm.analysis.logit_bias import (
    DAMAGE_LIMITS, NO_SHIFT_GRID, PRACTICAL_DELTA, TWO_WAY_GRID, bias_metrics, class_metrics, damage_limited_biases,
    selection_rules,
)
from mivlm.config import PREDICTIONS_DIR, TABLES_DIR
from mivlm.data.cases import image_linked_classes
from mivlm.evaluation.metrics import ONE_N_VARIANTS, load_rows
from mivlm.evaluation.report import interval_text, print_class_table

EXPERIMENT = "01_baseline"
MAIN_SETUP = "클래스 하나씩 (주 평가)"  # 판정에 쓰는 평가 방식의 이름. 나머지 둘은 민감도 분석이다
SLOT_LEVEL = 1 - 0.05 / len(ONE_N_VARIANTS)  # 세 자리를 같이 볼 때의 구간 (98.3%)
SHORT = {"original": "orig", "global": "glob", "class_cv": "cv", "class_oracle": "orac"}  # 표의 열 이름에 쓰는 줄임말
DIRECTION = {"지지": "나은 쪽", "반증": "나쁜 쪽", "미결정": "방향 불명"}  # direction_verdict의 결과를 읽는 말
PRACTICAL = {"앞 모델이 실질적으로 우월": "실질적으로 나음", "뒤 모델이 실질적으로 우월": "실질적으로 나쁨",
             "실질적 차이 없음": "실질적 차이 없음", "미결정": "미결정"}  # practical_verdict의 결과를 읽는 말
LIMITS = """[한계]
  탐색적 분석이다. 클래스별 margin의 차이를 이미 본 뒤에 설계했고, 독립된 확증이 아니다.
  기준 이동은 모델을 고치는 것이 아니라 답을 읽는 방식을 바꾸는 것이다. 이것으로 풀리는 몫이 커도 원인이 기준 하나라는 뜻은 아니다.
  class_cv는 그 클래스의 정답을 써서 기준을 정한다. 그룹끼리 이미지가 겹쳐서 낙관적일 수 있고, 독립된 일반화 결과가 아니다.
  class_oracle은 평가하는 케이스를 보고 기준을 정한 상한이다. 방법이 아니다.
  global과 class_cv의 차이는 쓰는 정보와 나누는 방식이 달라서 생긴 것이기도 하다. 클래스 성향의 인과적 효과로 읽지 않는다.
  bf16 실행의 margin은 0.125 간격이라, 손상 한도에 딱 맞는 기준이 없을 수 있다. 실제 손상률을 같이 본다.
  구간은 클래스를 단위로 다시 뽑아 만든다. 단위가 적어(클래스 20개) 실제보다 조금 좁을 수 있다.
  같은 이미지를 여러 케이스가 돌려 쓰고, 일부 이미지는 두 클래스가 같이 쓴다."""


def class_interval(by_class: dict[str, dict], key: str, level: float = 0.95) -> str:
    """key의 클래스 평균과 구간. 값을 정할 수 없는 클래스(nan)는 빼고, 뺐으면 쓴 클래스 수를 적는다"""
    per_class = [values[key] for values in by_class.values() if not math.isnan(values[key])]
    used = "" if len(per_class) == len(by_class) else f" ({len(per_class)}개)"
    return interval_text(*mean_interval(per_class, level)) + used


def print_rule_summary(title: str, rows: list[dict], rules: dict[str, list[float]]) -> None:
    """규칙마다: 클래스 평균의 지표와 구간, 그리고 케이스를 모두 합쳐 센 교정과 손상의 수"""
    print(f"\n[{title}]  클래스 평균 [95% 구간]")
    print(f"  {'':<14}{'균형 정확도':<28}{'PPP 정확도':<28}{'PPP 정확도 변화':<28}N 1장 중 False-Yes 비율")
    for rule, biases in rules.items():
        by_class = class_metrics(rows, biases)
        print(f"  {rule:<14}" + "".join(f"{class_interval(by_class, key):<30}" for key in
                                        ["balanced_accuracy", "ppp_accuracy", "ppp_accuracy_change", "false_yes_rate"]))
    print(f"  {'':<14}{'원래 False-Yes 중 교정':<34}원래 맞힌 PPP 중 손상   (케이스를 모두 합쳐 센 값)")
    for rule, biases in rules.items():
        pooled = bias_metrics(rows, biases)
        corrected = f"{pooled['n_corrected']}/{pooled['n_false_yes']} = {pooled['false_yes_correction_rate']:.1%}"
        damaged = f"{pooled['n_damaged']}/{pooled['n_ppp_yes']} = {pooled['ppp_damage_rate']:.1%}"
        print(f"  {rule:<14}{corrected:<36}{damaged}")


def comparison(by_class: dict[str, dict], baseline: dict[str, dict]) -> str:
    """클래스마다의 균형 정확도 차이(by_class - baseline)의 평균과 구간, 통계적 방향, 실질적 효과"""
    mean, low, high = mean_interval([by_class[name]["balanced_accuracy"] - baseline[name]["balanced_accuracy"]
                                     for name in by_class])
    return (f"{interval_text(mean, low, high):<28}{DIRECTION[direction_verdict(low, high)]:<10}"
            f"{PRACTICAL[practical_verdict(low, high, PRACTICAL_DELTA)]}")


def print_verdicts(title: str, rows_and_rules: dict[str, tuple]) -> None:
    """평가 방식마다 G, C의 판정과 상한(class_oracle - global)을 출력한다"""
    print(f"\n[{title}]  균형 정확도 차이의 클래스 평균 [95% 구간], 통계적 방향, 실질적 효과 (기준 {PRACTICAL_DELTA:.0%}p)")
    for setup, (rows, rules) in rows_and_rules.items():
        by_rule = {rule: class_metrics(rows, biases) for rule, biases in rules.items()}
        print(f"  평가 방식: {setup}")
        print(f"    {'G. global - original':<30}{comparison(by_rule['global'], by_rule['original'])}")
        print(f"    {'C. class_cv - global':<30}{comparison(by_rule['class_cv'], by_rule['global'])}")
        print(f"    {'상한. class_oracle - global':<30}{comparison(by_rule['class_oracle'], by_rule['global'])}  (판정 아님)")
    print("  C의 차이는 class_cv가 그 클래스의 정답을 쓰기 때문이기도 하다. 클래스 성향의 인과적 효과로 읽지 않는다.")


def print_damage_baseline(setup: str, rows: list[dict], fold_of: dict) -> None:
    """손상 한도마다: 다른 클래스에서 고른 기준, 평가한 쪽의 실제 손상률과 한도 위반 여부, False-Yes 교정"""
    print(f"\n[PPP 손상 한도의 기준선] 평가 방식: {setup}. 기준은 뺀 클래스 밖에서만 고른다")
    for limit in DAMAGE_LIMITS:
        biases = damage_limited_biases(rows, fold_of, limit)
        pooled, by_class = bias_metrics(rows, biases), class_metrics(rows, biases)
        over = sum(values["ppp_damage_rate"] > limit for values in by_class.values())
        print(f"  한도 {limit:.0%}: 고른 기준 {min(biases)} ~ {max(biases)}")
        print(f"    실제 손상 {pooled['n_damaged']}/{pooled['n_ppp_yes']} = {pooled['ppp_damage_rate']:.1%} "
              f"({'한도 위반' if pooled['ppp_damage_rate'] > limit else '한도 안'}), "
              f"클래스별로는 {over}개 클래스가 한도를 넘음. PPP 정확도 변화 {pooled['ppp_accuracy_change']:+.1%}p")
        print(f"    False-Yes 교정 {pooled['n_corrected']}/{pooled['n_false_yes']} = "
              f"{pooled['false_yes_correction_rate']:.1%}, N 1장 중 False-Yes 비율 {pooled['false_yes_rate']:.1%}")
        print(f"    클래스 평균: 손상률 {class_interval(by_class, 'ppp_damage_rate')}, "
              f"교정률 {class_interval(by_class, 'false_yes_correction_rate')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="결정 기준 이동 분석")
    parser.add_argument("run", help="outputs/predictions/01_baseline/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    rows = [row for row in load_rows(PREDICTIONS_DIR / EXPERIMENT / args.run)
            if row["variant"] in ["PPP", *ONE_N_VARIANTS]]
    pairs_by_class = class_pairs(rows)
    shared = shared_images(pairs_by_class)
    unshared_rows = [case for pairs in pairs_by_class.values() for pair in without_images(pairs, shared)
                     for case in pair["cases"].values()]
    one_class = {name: name for name in pairs_by_class}
    groups = image_linked_classes(rows)
    # 평가 방식 이름 -> (쓸 행들, 클래스를 어느 묶음으로 뺄지)
    setups = {
        MAIN_SETUP: (rows, one_class),
        "이어진 클래스를 같이 뺌": (rows, {name: number for number, group in enumerate(groups) for name in group}),
        "공유 이미지 pair 제외": (unshared_rows, one_class),
    }
    evaluated = {setup: (setup_rows, selection_rules(setup_rows, fold_of, NO_SHIFT_GRID))
                 for setup, (setup_rows, fold_of) in setups.items()}
    rules = evaluated[MAIN_SETUP][1]
    by_rule = {rule: class_metrics(rows, biases) for rule, biases in rules.items()}

    print(f"{'=' * 20} {args.run}: {RUN_ROLES.get(args.run, '역할을 정하지 않은 실행')} {'=' * 20}")
    print(f"케이스 {len(rows)}개 (PPP {sum(row['variant'] == 'PPP' for row in rows)}개, N 1장 "
          f"{sum(row['variant'] != 'PPP' for row in rows)}개), 클래스 {len(pairs_by_class)}개")
    mismatched = sum((row["prediction"] == "Yes") != (row["logit_margin"] > 0) for row in rows)
    print(f"[확인] b = 0의 답이 모델의 원래 답과 다른 케이스: {mismatched}개 (0이어야 한다). "
          f"margin이 정확히 0인 케이스 {sum(row['logit_margin'] == 0 for row in rows)}개는 No로 센다")
    print(f"후보 b: {NO_SHIFT_GRID[0]} ~ {NO_SHIFT_GRID[-1]}, 간격 {NO_SHIFT_GRID[1]} ({len(NO_SHIFT_GRID)}개). "
          f"가장 큰 margin {max(row['logit_margin'] for row in rows):.3f}")

    print_rule_summary("주 분석: No 쪽으로의 이동 (b >= 0), 평가 방식: " + MAIN_SETUP, rows, rules)
    print_verdicts("판정", evaluated)
    for setup in [MAIN_SETUP, "이어진 클래스를 같이 뺌"]:
        print_damage_baseline(setup, *setups[setup])

    print("\n[자리별 False-Yes 비율]  클래스 평균 [98.3% 구간]")
    print(f"  {'':<14}" + "".join(f"{variant:<30}" for variant in ONE_N_VARIANTS))
    for rule, biases in rules.items():
        cells = []
        for variant in ONE_N_VARIANTS:
            chosen = [(row, bias) for row, bias in zip(rows, biases) if row["variant"] == variant]
            by_class = class_metrics([row for row, _ in chosen], [bias for _, bias in chosen])
            cells.append(class_interval(by_class, "false_yes_rate", SLOT_LEVEL))
        print(f"  {rule:<14}" + "".join(f"{cell:<30}" for cell in cells))

    def mean_bias(name: str, biases: list[float]) -> float:
        chosen = [bias for row, bias in zip(rows, biases) if row["object"] == name]
        return sum(chosen) / len(chosen)

    table = {name: {**{f"b_{rule}": mean_bias(name, biases) for rule, biases in rules.items() if rule != "original"},
                    **{f"{key}_{rule}": by_rule[rule][name][key] for rule in rules
                       for key in ["balanced_accuracy", "ppp_accuracy", "false_yes_rate", "ppp_damage_rate",
                                   "false_yes_correction_rate"]}}
             for name in pairs_by_class}
    print_class_table("클래스별: 고른 기준 b (class_cv는 그룹 평균), False-Yes 비율(FY), PPP 정확도(PPP)", table, {
        "b glob": ("b_global", ".3f"), "b cv": ("b_class_cv", ".3f"), "b orac": ("b_class_oracle", ".3f"),
        **{f"FY {SHORT[rule]}": (f"false_yes_rate_{rule}", ".0%") for rule in rules},
        **{f"PPP {SHORT[rule]}": (f"ppp_accuracy_{rule}", ".0%") for rule in rules},
    })

    two_way = selection_rules(rows, one_class, TWO_WAY_GRID)
    print_rule_summary("별도 분석: 양방향 보정 (b가 음수도 될 수 있음), 평가 방식: " + MAIN_SETUP, rows, two_way)
    print_verdicts("별도 분석의 비교: 양방향 보정. 주 판정이 아니다", {MAIN_SETUP: (rows, two_way)})
    print(f"\n{LIMITS}")

    table_path = TABLES_DIR / EXPERIMENT / f"{args.run}_logit_bias.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        keys = list(next(iter(table.values())))
        writer.writerow(["object", *keys])
        for name, values in table.items():
            writer.writerow([name, *[values[key] for key in keys]])
    print(f"\n[ok] {table_path}")


if __name__ == "__main__":
    main()
