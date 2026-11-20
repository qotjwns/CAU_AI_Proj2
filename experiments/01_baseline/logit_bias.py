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
클래스별 손상 한도의 기준선 (8-A): 같은 한도를 같은 클래스 안에서 건다. class_cv는 평가 그룹을 뺀 나머지 그룹에서,
    class_oracle(참고 상한)은 클래스 전체에서 고른다. 학습 쪽에 원래 맞힌 PPP가 없으면 b = 0을 쓰고 그 수를 적는다.

G1 (Calibration 한계, 탐색적 관문. 기존 class_cv 결과를 일부 본 뒤에 정했다). 주 판정은 3B bf16 실행이다
    5% 한도의 class_cv에서, 평가 쪽 실제 PPP 손상(합친 값)이 5% 이하이고 합친 False-Yes 교정률의 95% 구간 상한이
    50%보다 작으면 통과, 하한이 50% 이상이면 실패. 평가 쪽 손상이 5%를 넘으면 보류(손상 예산 위반), 그 밖에도 보류.
    통과는 이 규칙(제약형 class_cv)의 한계이지, 가능한 모든 보정 방법의 한계가 아니다.

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
    DAMAGE_LIMITS, G1_DAMAGE_LIMIT, PRACTICAL_DELTA, bias_metrics, class_damage_limited_biases, class_metrics,
    damage_limited_biases, g1_verdict, no_shift_candidates, pooled_rate_interval, selection_rules, two_way_candidates,
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
  후보 b는 c9c2899까지 0.125 간격의 격자였고, float32 실행의 손상 한도 2%, 5% 기준선이 그 격자 때문에 b = 0으로 떨어졌다.
  결과를 본 뒤에 후보를 margin 값으로 바꿨으므로, 격자 결과와 이 결과는 민감도 분석으로 같이 적는다.
  클래스별 손상 한도(8-A)에서 학습 쪽의 맞힌 PPP는 클래스마다 십여 개뿐이다. 그래서 5% 한도는 사실상 학습 쪽 손상 0개를 뜻하고,
  그룹(맞힌 PPP 0~4개)에서는 한 개만 손상돼도 한도를 넘는다. 그룹별 위반은 이산성 때문에 많아 보일 수 있다.
  G1은 기존 class_cv 결과를 일부 본 뒤에 정한 탐색적 관문이다. class_oracle은 평가 라벨을 쓴 참고 상한이라 판정이 없다.
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


def print_class_damage_baseline(run: str, rows: list[dict]) -> None:
    """클래스별 손상 한도의 기준선(8-A): 한도마다 class_cv와 class_oracle의 기준, 평가 쪽 손상과 교정, 그리고 G1 판정"""
    print("\n[클래스별 PPP 손상 한도의 기준선 (8-A)] 기준은 같은 클래스 안에서만 고른다. "
          "class_cv는 평가 그룹을 뺀 나머지 그룹에서, class_oracle(참고 상한, 평가 라벨 사용)은 클래스 전체에서")
    units = {"class_cv": len({(row["object"], row["group_id"]) for row in rows}),
             "class_oracle": len({row["object"] for row in rows})}
    g1 = None
    for limit in DAMAGE_LIMITS:
        biases_by_rule, fallback_units = class_damage_limited_biases(rows, limit)
        print(f"  한도 {limit:.0%}:")
        for rule, biases in biases_by_rule.items():
            pooled, by_class = bias_metrics(rows, biases), class_metrics(rows, biases)
            # 그룹(평가 단위)마다의 실제 손상. 원래 맞힌 PPP가 없는 그룹은 손상률을 정할 수 없어 세지 않는다
            by_group = {}
            for row, bias in zip(rows, biases):
                by_group.setdefault((row["object"], row["group_id"]), []).append((row, bias))
            group_damage = [bias_metrics([row for row, _ in chosen], [bias for _, bias in chosen])
                            for chosen in by_group.values()]
            groups_with_ppp = [values for values in group_damage if values["n_ppp_yes"] > 0]
            correction, low, high = pooled_rate_interval([(values["n_corrected"], values["n_false_yes"])
                                                          for values in by_class.values()])
            over = pooled["ppp_damage_rate"] > limit
            print(f"    {rule}: 고른 기준 {min(biases):.3f} ~ {max(biases):.3f}, "
                  f"학습 쪽에 맞힌 PPP가 없어 b = 0을 쓴 단위 {fallback_units[rule]}/{units[rule]}개")
            print(f"      실제 손상 {pooled['n_damaged']}/{pooled['n_ppp_yes']} = {pooled['ppp_damage_rate']:.1%} "
                  f"({'한도 위반' if over else '한도 안'}), 한도를 넘은 클래스 "
                  f"{sum(values['ppp_damage_rate'] > limit for values in by_class.values())}개, "
                  f"한도를 넘은 그룹 {sum(values['ppp_damage_rate'] > limit for values in groups_with_ppp)}"
                  f"/{len(groups_with_ppp)}개 (맞힌 PPP가 있는 그룹)")
            print(f"      False-Yes 교정 {pooled['n_corrected']}/{pooled['n_false_yes']} = "
                  f"{interval_text(correction, low, high)} (클래스를 다시 뽑은 95% 구간), "
                  f"N 1장 중 False-Yes 비율 {pooled['false_yes_rate']:.1%}")
            if rule == "class_cv" and limit == G1_DAMAGE_LIMIT:
                g1 = (pooled["ppp_damage_rate"], correction, low, high)
    damage, correction, low, high = g1
    print(f"\n[G1] 5% 한도의 class_cv: 평가 쪽 실제 손상 {damage:.1%}, 합친 교정률 {interval_text(correction, low, high)}"
          f" -> {g1_verdict(damage, low, high)}")
    print(f"  이 실행의 역할: {RUN_ROLES.get(run, '역할을 정하지 않은 실행')}. G1의 주 판정은 3B bf16(qwen2.5-vl-3b)이다.")


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
    evaluated = {setup: (setup_rows, selection_rules(setup_rows, fold_of, no_shift_candidates))
                 for setup, (setup_rows, fold_of) in setups.items()}
    rules = evaluated[MAIN_SETUP][1]
    by_rule = {rule: class_metrics(rows, biases) for rule, biases in rules.items()}

    print(f"{'=' * 20} {args.run}: {RUN_ROLES.get(args.run, '역할을 정하지 않은 실행')} {'=' * 20}")
    print(f"케이스 {len(rows)}개 (PPP {sum(row['variant'] == 'PPP' for row in rows)}개, N 1장 "
          f"{sum(row['variant'] != 'PPP' for row in rows)}개), 클래스 {len(pairs_by_class)}개")
    mismatched = sum((row["prediction"] == "Yes") != (row["logit_margin"] > 0) for row in rows)
    print(f"[확인] b = 0의 답이 모델의 원래 답과 다른 케이스: {mismatched}개 (0이어야 한다). "
          f"margin이 정확히 0인 케이스 {sum(row['logit_margin'] == 0 for row in rows)}개는 No로 센다")
    print(f"후보 b: 0과, 기준을 고르는 쪽 행들의 margin 값 (평가하는 쪽의 margin은 쓰지 않는다). "
          f"전체 행의 후보는 {len(no_shift_candidates(rows))}개, 가장 큰 margin {max(row['logit_margin'] for row in rows):.3f}")

    print_rule_summary("주 분석: No 쪽으로의 이동 (b >= 0), 평가 방식: " + MAIN_SETUP, rows, rules)
    print_verdicts("판정", evaluated)
    for setup in [MAIN_SETUP, "이어진 클래스를 같이 뺌"]:
        print_damage_baseline(setup, *setups[setup])
    print_class_damage_baseline(args.run, rows)

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

    two_way = selection_rules(rows, one_class, two_way_candidates)
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
