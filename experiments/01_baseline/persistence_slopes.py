"""
PPP 증거가 1N에 남는 정도(7-A-1): 같은 pair의 1N margin을 PPP margin에 회귀한 기울기, 자리(변형)마다
python experiments/01_baseline/persistence_slopes.py qwen2.5-vl-3b            # 주 분석
python experiments/01_baseline/persistence_slopes.py qwen2.5-vl-7b            # 모델 견고성 분석
python experiments/01_baseline/persistence_slopes.py qwen2.5-vl-3b_float32    # 정밀도 민감도 분석

저장된 A(01_baseline) 결과를 읽기만 한다. 모델은 쓰지 않는다. 표는 터미널에만 출력한다.
기울기의 정의(전체 회귀, 클래스 내부 회귀)는 mivlm.analysis.position_weights 참고.

이 분석은 판정을 내리지 않는다. 새 이미지 풀(독립 확증)에서 쓸 예측값과 판정 영역을 정하기 위한 탐색적 추정이다.
구간: 클래스를 단위로 10,000번 다시 뽑은 백분위 구간 (seed 고정). 기울기는 95%,
    자리끼리의 차이는 세 쌍을 같이 보므로 98.3%이다.
"""
import argparse
from itertools import combinations

from mivlm.analysis.class_sensitivity import RUN_ROLES, class_bootstrap_interval, class_pairs
from mivlm.analysis.position_weights import slope_sums, slopes
from mivlm.config import PREDICTIONS_DIR
from mivlm.evaluation.metrics import ONE_N_VARIANTS, load_rows
from mivlm.evaluation.report import interval_text, print_class_table

EXPERIMENT = "01_baseline"
N_SLOT = {variant: variant.index("N") + 1 for variant in ONE_N_VARIANTS}  # 변형마다 N이 들어간 자리 (PPN은 3)
DIFFERENCES = list(combinations(range(len(ONE_N_VARIANTS)), 2))  # 자리끼리의 차이: (앞 변형, 뒤 변형)의 위치
DIFFERENCE_LEVEL = 1 - 0.05 / len(DIFFERENCES)  # 세 쌍을 같이 보므로 구간을 넓힌다 (98.3%)
LIMITS = """[한계]
  탐색적 추정이다. 같은 데이터로 한 즉석 계산을 본 뒤에 정의했고, 독립된 확증이 아니다.
  기울기는 PPP margin의 변동 중 1N margin에 남는 몫이다. 가산 결합이면 남은 두 자리가 PPP 변동에서 차지하는 몫으로
  정해지므로, 기울기만으로 AND 여부나 모델 내부의 집계 방식을 판단하지 않는다. 0이 곧 AND라는 뜻도 아니다.
  전체 회귀는 클래스 사이의 차이(클래스 성향)를 포함하고, 클래스 내부 회귀는 그것을 뺀다.
  클래스 내부 회귀에서는 x의 변동이 큰 클래스가 더 큰 무게를 갖는다. 클래스별 기울기는 pair 20개로 낸 값이라 흔들림이 크다.
  같은 이미지를 여러 pair가 돌려 쓰고, 일부 이미지는 두 클래스가 같이 쓴다. 구간은 클래스를 단위로 다시 뽑아 만든다.
  단위가 적어(클래스 20개) 구간이 실제보다 조금 좁을 수 있다.
  세 실행은 같은 케이스를 돌린 것이라, 값이 비슷해도 독립된 재현이 아니다."""


def main() -> None:
    parser = argparse.ArgumentParser(description="PPP 증거가 1N에 남는 정도: 자리별 기울기")
    parser.add_argument("run", help="outputs/predictions/01_baseline/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    pairs_by_class = class_pairs(load_rows(PREDICTIONS_DIR / EXPERIMENT / args.run))
    # 회귀 이름 -> 클래스마다의 충분통계 (클래스 수, 변형 3개, 5)
    sums = {"전체 회귀": slope_sums(pairs_by_class, centered=False),
            "클래스 내부 회귀": slope_sums(pairs_by_class, centered=True)}

    print(f"{'=' * 20} {args.run}: {RUN_ROLES.get(args.run, '역할을 정하지 않은 실행')} {'=' * 20}")
    print(f"pair {sum(len(pairs) for pairs in pairs_by_class.values())}개, 클래스 {len(pairs_by_class)}개. "
          f"x = PPP margin, y = 같은 pair의 변형 margin")

    print("\n[자리별 기울기]  값 [95% 구간]")
    print(f"  {'변형':<6}{'N 자리':<8}" + "".join(f"{name:<30}" for name in sums))
    estimates = {name: slopes(values) for name, values in sums.items()}
    intervals = {name: class_bootstrap_interval(values, slopes, 0.95) for name, values in sums.items()}
    for position, variant in enumerate(ONE_N_VARIANTS):
        cells = [interval_text(estimates[name][position].item(), *intervals[name][:, position].tolist())
                 for name in sums]
        print(f"  {variant:<8}{N_SLOT[variant]:<10}" + "".join(f"{cell:<32}" for cell in cells))

    def differences(slope_values):
        """기울기 (..., 변형 3개) -> 자리끼리의 차이 (..., 쌍 3개). 쌍의 순서는 DIFFERENCES"""
        return slope_values[..., [a for a, _ in DIFFERENCES]] - slope_values[..., [b for _, b in DIFFERENCES]]

    print(f"\n[자리끼리의 기울기 차이]  앞 변형 - 뒤 변형, 값 [{DIFFERENCE_LEVEL:.1%} 구간]")
    print(f"  {'쌍':<14}" + "".join(f"{name:<30}" for name in sums))
    difference_estimates = {name: differences(estimates[name]) for name in sums}
    difference_intervals = {name: class_bootstrap_interval(values, lambda drawn: differences(slopes(drawn)),
                                                           DIFFERENCE_LEVEL)
                            for name, values in sums.items()}
    for position, (a, b) in enumerate(DIFFERENCES):
        label = f"{ONE_N_VARIANTS[a]} - {ONE_N_VARIANTS[b]}"
        cells = [interval_text(difference_estimates[name][position].item(),
                               *difference_intervals[name][:, position].tolist()) for name in sums]
        print(f"  {label:<16}" + "".join(f"{cell:<32}" for cell in cells))

    # 클래스 하나만으로 낸 기울기. 한 클래스 안에서는 전체 회귀와 클래스 내부 회귀가 같다
    within = sums["클래스 내부 회귀"]
    table = {name: {variant: slopes(within[position:position + 1])[index].item()
                    for index, variant in enumerate(ONE_N_VARIANTS)}
             for position, name in enumerate(pairs_by_class)}
    print_class_table("클래스별 기울기 (그 클래스의 pair만으로 낸 값. 구간 없음)", table,
                      {variant: (variant, "+.2f") for variant in ONE_N_VARIANTS})
    print(f"\n{LIMITS}")


if __name__ == "__main__":
    main()
