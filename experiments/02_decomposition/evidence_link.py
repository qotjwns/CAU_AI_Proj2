"""
단일 이미지 margin 연계 분석: P를 N으로 바꿨을 때의 공동 margin 하락을, 이미지를 한 장씩 물은 margin으로 예측한다
python experiments/02_decomposition/evidence_link.py qwen2.5-vl-3b    # 주 분석
python experiments/02_decomposition/evidence_link.py qwen2.5-vl-7b    # 모델 견고성 분석

저장된 A(01_baseline)와 B(02_decomposition/b_single_image) 결과를 읽기만 한다. 모델은 쓰지 않는다.
표는 터미널에 출력하고, 클래스별 오차는 outputs/tables/02_decomposition/<실행 이름>_evidence_link.csv에도 저장한다.
관측, 설명 변수, 모델의 정의는 mivlm.analysis.evidence_link 참고. 3B float32는 한 장씩 물은 결과가 없어 대상이 아니다.
P→P 대조군 분석(experiments/01_baseline/replacement_control.py)과는 판정을 따로 관리하고, 그 결과를 읽지 않는다.

평가: 클래스 하나를 빼고 나머지 클래스에서 계수를 맞춘 뒤 뺀 클래스를 예측한다. 계수와 보정은 학습 클래스에서만 정한다.
    오차는 평균 절대 오차(MAE)이고, 클래스마다 하나씩 낸다. 제곱 오차는 판정에 쓰지 않는다.
비교: 두 모델의 차이 = 뒤 모델의 MAE - 앞 모델의 MAE (클래스마다 짝지어 계산). 양수면 앞 모델이 더 정확하다.
    구간은 클래스를 단위로 10,000번 다시 뽑아 만든다 (seed 고정, 95%).

판정 기준 (결과를 보기 전에 정했다. 결과를 본 뒤에 바꾸지 않는다)
    주 비교는 min+slot 대 mean+slot의 하락 예측이다. min 대 mean은 보조 비교, 공동 margin 자체의 예측은 보조 분석이다.
    통계적 방향: 차이의 구간 전체가 0보다 크면 앞 모델 쪽, 전체가 0보다 작으면 뒤 모델 쪽, 그 밖에는 방향 불명
    실질적 효과: delta = 절편만 쓴 모델의 MAE(클래스 평균)의 5%. 같은 실행, 같은 대상, 같은 평가 방식의 값으로 계산한다
        구간 전체가 +delta를 넘으면 앞 모델이, 전체가 -delta 아래면 뒤 모델이 실질적으로 우월
        구간 전체가 -delta ~ +delta 안이면 실질적 차이 없음, 그 밖에는 미결정
    두 판정은 따로 적는다. 방향이 있어도 delta를 넘지 않으면 우월하다고 적지 않는다.
    절편 모델이나 자리 모델보다 나은지, P 증거가 N만 쓴 모델에 더하는 것이 있는지는 같은 방식으로 따로 보고한다.
"""
import argparse
import csv
import math
import statistics

import torch

from mivlm.analysis.class_sensitivity import (
    RUN_ROLES, class_pairs, direction_verdict, mean_interval, rank_correlation, shared_images, without_images,
)
from mivlm.analysis.evidence_link import (
    DELTA_SHARE, DROP_MODELS, LEVEL_MODELS, class_mean, drop_observations, level_observations, n_strength_bins,
    out_of_fold_predictions, practical_verdict, with_single_margins,
)
from mivlm.config import TABLES_DIR
from mivlm.data.cases import image_linked_classes
from mivlm.evaluation.decomposition import load_case_rows
from mivlm.evaluation.metrics import ONE_N_VARIANTS
from mivlm.evaluation.report import interval_text, print_class_table

EXPERIMENT = "02_decomposition"
SLOT_LEVEL = 1 - 0.05 / len(ONE_N_VARIANTS)  # 세 자리를 같이 볼 때의 구간 (98.3%)
MAIN_SETUP = "클래스 하나씩 (주 평가)"  # 판정에 쓰는 평가 방식의 이름. 나머지 둘은 민감도 분석이다
DIRECTION = {"지지": "앞 모델 쪽", "반증": "뒤 모델 쪽", "미결정": "방향 불명"}  # direction_verdict의 결과를 읽는 말
# (묶음 이름, 앞 모델, 뒤 모델). 차이 = 뒤 모델의 MAE - 앞 모델의 MAE
COMPARISONS = [
    ("주 비교", "min+slot", "mean+slot"),
    ("보조 비교", "min", "mean"),
    ("절편 모델 대비", "slot", "intercept"), ("절편 모델 대비", "n", "intercept"),
    ("절편 모델 대비", "min", "intercept"), ("절편 모델 대비", "mean", "intercept"),
    ("자리 모델 대비", "n+slot", "slot"), ("자리 모델 대비", "min+slot", "slot"), ("자리 모델 대비", "mean+slot", "slot"),
    ("P 증거의 추가 가치", "min+slot", "n+slot"), ("P 증거의 추가 가치", "mean+slot", "n+slot"),
    ("P 증거의 추가 가치", "min", "n"), ("P 증거의 추가 가치", "mean", "n"),
]
LIMITS = """[한계]
  탐색적 분석이다. 기존 P→N 결과를 본 뒤에 설계했고, 독립된 확증이 아니다.
  여기서 비교하는 것은 예측 규칙의 정확도다. min을 쓴 규칙이 더 정확해도 모델이 내부에서 최솟값을 계산한다는 뜻이 아니고,
  mean을 쓴 규칙이 더 정확해도 평균을 계산한다는 뜻이 아니다. 내부의 집계 방식은 이 분석으로 알 수 없다.
  단일 margin과 공동 margin은 질문이 달라 직접 비교하지 않았다. 단일 margin은 학습 클래스에서 맞춘 계수를 거쳐서만 썼다.
  N의 단일 margin이 세 장 중 가장 낮은 관측이 많을수록 min과 mean은 N 부분을 공유하고 P 부분만 달라진다 (위의 설명 변수 점검).
  delta는 절편 모델의 오차에서 추정한 값이라 그 자체에 불확실성이 있고, 5%라는 비율은 연구자가 정한 탐색적 기준이다.
  클래스별 오차는 학습 클래스가 서로 겹쳐서 완전히 독립이 아니고, 단위가 클래스 20개라 구간이 조금 좁을 수 있다.
  같은 이미지를 여러 케이스가 돌려 쓰고, 일부 이미지는 두 클래스가 같이 쓴다."""


def class_errors(observations: list[dict], predictions: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """모델마다 클래스별 평균 절대 오차: {모델: {클래스: MAE}}"""
    return {model: class_mean(observations, [abs(observation["target"] - value)
                                             for observation, value in zip(observations, predicted)])
            for model, predicted in predictions.items()}


def print_model_errors(title: str, errors_by_setup: dict[str, dict]) -> None:
    """모델마다 MAE의 클래스 평균과 95% 구간을 평가 방식마다 나란히 출력한다"""
    print(f"\n[{title}]  MAE의 클래스 평균 [95% 구간]. 열: {' / '.join(errors_by_setup)}")
    for model in next(iter(errors_by_setup.values())):
        cells = [interval_text(*mean_interval(list(errors[model].values()))) for errors in errors_by_setup.values()]
        print(f"  {model:<12}" + "".join(f"{cell:<30}" for cell in cells))


def print_comparisons(title: str, errors_by_setup: dict[str, dict], models: dict) -> None:
    """평가 방식마다: delta와, COMPARISONS의 비교마다 차이의 클래스 평균, 구간, 통계적 방향, 실질적 효과"""
    for setup, errors in errors_by_setup.items():
        intercept_mean, intercept_low, intercept_high = mean_interval(list(errors["intercept"].values()))
        delta = DELTA_SHARE * intercept_mean
        print(f"\n[{title}] 평가 방식: {setup}")
        print(f"  delta {delta:.3f} = 절편 모델의 MAE {intercept_mean:.3f}의 {DELTA_SHARE:.0%}. "
              f"그 MAE의 95% 구간으로 보면 {DELTA_SHARE * intercept_low:.3f} ~ {DELTA_SHARE * intercept_high:.3f}. "
              f"판정에는 {delta:.3f}만 쓴다")
        print(f"  {'':<20}{'앞 모델 vs 뒤 모델':<24}{'뒤의 MAE - 앞의 MAE [95% 구간]':<32}{'통계적 방향':<12}실질적 효과")
        for label, first, second in COMPARISONS:
            if first in models and second in models:
                mean, low, high = mean_interval([errors[second][name] - errors[first][name] for name in errors[first]])
                print(f"  {label:<20}{f'{first} vs {second}':<24}{interval_text(mean, low, high):<32}"
                      f"{DIRECTION[direction_verdict(low, high)]:<12}{practical_verdict(low, high, delta)}")


def print_rank_correlations(observations: list[dict], predictions: dict[str, list[float]]) -> None:
    """클래스 안에서, 예측한 값과 관측한 값의 순위 상관. 모델마다 클래스 평균과 95% 구간 (판정 없음)"""
    indices_by_class = {}
    for index, observation in enumerate(observations):
        indices_by_class.setdefault(observation["object"], []).append(index)
    target = torch.tensor([observation["target"] for observation in observations], dtype=torch.float64)
    print("\n[클래스 안에서의 순위 상관: 예측한 하락 대 관측한 하락]  클래스 평균 [95% 구간]. 판정 없음")
    for model, predicted in predictions.items():
        predicted = torch.tensor(predicted, dtype=torch.float64)
        per_class = [rank_correlation(predicted[indices], target[indices]).item()
                     for indices in indices_by_class.values()]
        per_class = [value for value in per_class if not math.isnan(value)]
        if per_class:
            print(f"  {model:<12}{interval_text(*mean_interval(per_class))}  (클래스 {len(per_class)}개)")
        else:
            print(f"  {model:<12}클래스 안에서 예측값이 모두 같아 정할 수 없음")


def print_slot_residuals(observations: list[dict], predictions: dict[str, list[float]]) -> None:
    """자리를 쓰지 않는 모델의 잔차(관측 - 예측)를 자리별로. 클래스 평균과 98.3% 구간 (판정 없음)"""
    print("\n[자리별 잔차: 관측한 하락 - 예측한 하락]  클래스 평균 [98.3% 구간]. 판정 없음")
    print(f"  {'':<12}" + "".join(f"{variant:<30}" for variant in ONE_N_VARIANTS))
    for model, (_, by_slot) in DROP_MODELS.items():
        if not by_slot:
            cells = []
            for variant in ONE_N_VARIANTS:
                chosen = [(observation, observation["target"] - value)
                          for observation, value in zip(observations, predictions[model])
                          if observation["slot"] == variant]
                per_class = class_mean([observation for observation, _ in chosen], [residual for _, residual in chosen])
                cells.append(interval_text(*mean_interval(list(per_class.values()), SLOT_LEVEL)))
            print(f"  {model:<12}" + "".join(f"{cell:<30}" for cell in cells))


def print_false_yes_by_n_strength(observations: list[dict]) -> None:
    """클래스 안에서 N의 단일 margin을 세 구간으로 나눠, 구간별 False-Yes 비율의 클래스 평균과 95% 구간 (판정 없음)"""
    print("\n[N의 단일 margin 구간별 False-Yes 비율]  클래스 안에서 N 이미지를 셋으로 나눔. 클래스 평균 [95% 구간]. 판정 없음")
    bins = n_strength_bins(observations)
    for number, label in enumerate(["가장 강하게 No인 3분의 1", "가운데 3분의 1", "가장 약하게 No인 3분의 1"]):
        chosen = [observation for observation, bin_number in zip(observations, bins) if bin_number == number]
        per_class = class_mean(chosen, [float(observation["false_yes"]) for observation in chosen])
        print(f"  {label:<26}{interval_text(*mean_interval(list(per_class.values())))}  (관측 {len(chosen)}개)")


def main() -> None:
    parser = argparse.ArgumentParser(description="단일 이미지 margin 연계 분석")
    parser.add_argument("run", help="01_baseline과 02_decomposition/b_single_image 아래의 실행 폴더 이름")
    args = parser.parse_args()

    case_rows, _, _ = load_case_rows(args.run)
    rows = [row for row in with_single_margins(case_rows["A"], case_rows["B"])
            if row["variant"] in ["PPP", *ONE_N_VARIANTS]]
    pairs_by_class = class_pairs(rows)
    one_class = {name: name for name in pairs_by_class}
    groups = image_linked_classes(rows)
    shared = shared_images(pairs_by_class)
    unshared_pairs = {name: without_images(pairs, shared) for name, pairs in pairs_by_class.items()}
    # 평가 방식 이름 -> (쓸 pair들, 클래스를 어느 묶음으로 뺄지)
    setups = {
        MAIN_SETUP: (pairs_by_class, one_class),
        "이어진 클래스를 같이 뺌": (pairs_by_class, {name: number for number, group in enumerate(groups) for name in group}),
        "공유 이미지 pair 제외": (unshared_pairs, one_class),
    }

    # 주 분석(하락)과 보조 분석(공동 margin 자체)을 평가 방식마다 계산한다
    drop_errors, level_errors = {}, {}
    for setup, (source, fold_of) in setups.items():
        drops = [observation for pairs in source.values() for observation in drop_observations(pairs)]
        levels = [observation for pairs in source.values() for observation in level_observations(pairs)]
        drop_errors[setup] = class_errors(drops, out_of_fold_predictions(drops, fold_of, DROP_MODELS))
        level_errors[setup] = class_errors(levels, out_of_fold_predictions(levels, fold_of, LEVEL_MODELS))
    observations = [observation for pairs in pairs_by_class.values() for observation in drop_observations(pairs)]
    predictions = out_of_fold_predictions(observations, one_class, DROP_MODELS)
    main_errors, main_level_errors = drop_errors[MAIN_SETUP], level_errors[MAIN_SETUP]

    print(f"{'=' * 20} {args.run}: {RUN_ROLES.get(args.run, '역할을 정하지 않은 실행')} {'=' * 20}")
    print(f"클래스 {len(pairs_by_class)}개, 관측 {len(observations)}개 ((pair, 변형)). "
          f"공유 이미지로 이어진 클래스의 덩어리 {len(groups)}개 "
          f"(크기 {sorted((len(group) for group in groups), reverse=True)})")
    print(f"두 클래스가 같이 쓰는 이미지 {len(shared)}장. 그 이미지가 든 pair를 빼면 "
          f"pair {sum(len(pairs) for pairs in unshared_pairs.values())}개가 남는다")
    print("\n[설명 변수 점검]")
    print(f"  N의 단일 margin이 세 장 중 가장 낮은 관측: {statistics.mean(o['n_is_lowest'] for o in observations):.1%}")
    print(f"  min과 mean의 상관: "
          f"{statistics.correlation([o['min'] for o in observations], [o['mean'] for o in observations]):.3f}")

    print_model_errors("주 분석: 하락의 예측 오차", drop_errors)
    print_comparisons("주 분석: 하락의 예측, 모델 비교", drop_errors, DROP_MODELS)
    print_class_table("클래스별 MAE (주 평가). d main = MAE(mean+slot) - MAE(min+slot), d plain = MAE(mean) - MAE(min)", {
        name: {**{model: main_errors[model][name] for model in DROP_MODELS},
               "d_main": main_errors["mean+slot"][name] - main_errors["min+slot"][name],
               "d_plain": main_errors["mean"][name] - main_errors["min"][name]}
        for name in pairs_by_class
    }, {**{model: (model, ".2f") for model in DROP_MODELS},
        "d main": ("d_main", "+.3f"), "d plain": ("d_plain", "+.3f")})
    print_rank_correlations(observations, predictions)
    print_slot_residuals(observations, predictions)
    print_false_yes_by_n_strength(observations)

    print_model_errors("보조 분석: 공동 margin 자체의 예측 오차 (PPP와 세 변형, 자리 = 변형)", level_errors)
    print_comparisons("보조 분석: 공동 margin 자체의 예측, 모델 비교", level_errors, LEVEL_MODELS)
    print(f"\n{LIMITS}")

    table_path = TABLES_DIR / EXPERIMENT / f"{args.run}_evidence_link.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["object", *[f"drop_mae_{model}" for model in DROP_MODELS],
                         *[f"level_mae_{model}" for model in LEVEL_MODELS]])
        for name in pairs_by_class:
            writer.writerow([name, *[main_errors[model][name] for model in DROP_MODELS],
                             *[main_level_errors[model][name] for model in LEVEL_MODELS]])
    print(f"\n[ok] {table_path}")


if __name__ == "__main__":
    main()
