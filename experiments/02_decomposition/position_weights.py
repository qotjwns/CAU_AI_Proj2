"""
자리 가중치와 N 하락 몫(7-A-2): N이 만드는 공동 margin의 하락이 자리 사이에 나뉘는 비율이,
PPP의 자리 가중치와 단일 증거의 변화량으로 계산한 가산 모형의 예측과 같은가
python experiments/02_decomposition/position_weights.py qwen2.5-vl-3b    # 주 분석
python experiments/02_decomposition/position_weights.py qwen2.5-vl-7b    # 모델 견고성 분석

저장된 A(01_baseline)와 B(02_decomposition/b_single_image) 결과를 읽기만 한다. 모델은 쓰지 않는다. 표는 터미널에만 출력한다.
3B float32는 한 장씩 물은 결과가 없어 대상이 아니다. 기호와 정의는 mivlm.analysis.position_weights 참고.

판정 기준 (이 분석을 실행하기 전에 정했다. 결과를 본 뒤에 바꾸지 않는다)
    주 지표: 자리 k마다 r_k = δ_k - δ̂_k (관측 하락 몫 - 가산 모형의 예측 몫)
    구간: 클래스를 단위로 10,000번 다시 뽑은 백분위 구간. 세 자리를 같이 보므로 98.3%다 (Bonferroni)
    구간 전체가 [-0.05, +0.05] 안이면 실질적 동등, 하한이 +0.05보다 크면 양의 편차, 상한이 -0.05보다 작으면 음의 편차,
        그 밖에는 보류. 0.05(하락 몫 5%p)는 연구자가 정한 탐색적 기준이고, 이론에서 나온 값이 아니다
    세 자리가 모두 실질적 동등이면 "가산 모형과 실질적 동등성 지지"로 적는다
    보조 지표(δ - ω, ρ, 기울기)는 같은 98.3% 구간으로 내고 판정을 붙이지 않는다
"""
import argparse

import torch

from mivlm.analysis.class_sensitivity import N_DRAWS, RUN_ROLES, class_bootstrap_interval, class_pairs
from mivlm.analysis.evidence_link import practical_verdict, with_single_margins
from mivlm.analysis.position_weights import SLOT_VARIANTS, weight_statistics, weight_terms
from mivlm.evaluation.decomposition import load_case_rows
from mivlm.evaluation.metrics import ONE_N_VARIANTS
from mivlm.evaluation.report import interval_text

LEVEL = 1 - 0.05 / len(SLOT_VARIANTS)  # 세 자리를 같이 보므로 구간을 넓힌다 (98.3%)
EQUIVALENCE_DELTA = 0.05  # 실질적 동등의 범위: 하락 몫 ±5%p. 연구자가 정한 탐색적 기준이다
VERDICT = {"앞 모델이 실질적으로 우월": "양의 편차", "뒤 모델이 실질적으로 우월": "음의 편차",
           "실질적 차이 없음": "실질적 동등", "미결정": "보류"}  # practical_verdict의 결과를 r_k에 맞게 읽는 말
# 표의 묶음: (제목, [(weight_statistics의 이름, 줄 이름), ...])
GROUPS = [
    ("PPP의 자리 가중치", [("weights", "w"), ("weight_share", "ω = w / Σw")]),
    ("N 하락의 자리별 몫 (주 판정)", [("drop", "D 공동 margin 하락"), ("change", "Δ 단일 margin 변화"),
                              ("drop_share", "δ = D / ΣD"), ("predicted_share", "δ̂ = wΔ / ΣwΔ"),
                              ("share_gap", "r = δ - δ̂ (주 지표)")]),
    ("보조 지표 (판정 없음)", [("share_vs_weight", "δ - ω"), ("magnitude_ratio", "ρ = D / (wΔ)"),
                       ("kept_slope", "기울기: 남은 P만"), ("additive_slope", "기울기: 가산 모형"),
                       ("observed_slope", "기울기: 관측"), ("slope_gap", "관측 - 가산 모형")]),
]
NAMES = [name for _, rows in GROUPS for name, _ in rows]
LIMITS = """[한계]
  탐색적 분석이다. 같은 데이터의 기울기와 가중치를 즉석 계산으로 본 뒤에 설계했고, 독립된 확증이 아니다.
  δ̂은 N의 단일 margin으로 계산한다. 가중치는 P 이미지들의 변동에서 추정했으므로, N은 그 범위 밖으로의 외삽이다.
  모든 자리에 공통인 눈금 차이는 몫에서 상쇄되지만, 자리마다 다른 비선형성(포화)은 상쇄되지 않는다.
  그래서 r의 편차는 "그 자리의 N이 덜(더) 반영된다"뿐 아니라 "자리마다 비선형성이 다르다"로도 읽힐 수 있다.
  세 r의 합은 0이다. 한 자리의 양의 편차는 다른 자리의 음의 편차와 같이 나타나는 상대적 비교다.
  단일 margin의 잡음은 w를 줄인다(감쇠). 몫에서는 상쇄되지만 ρ에는 남는다. ρ는 크기 비교라 판정에 쓰지 않는다.
  w는 클래스 합으로, D와 Δ는 클래스 평균으로 무게를 준다.
  단일 질문(B)과 공동 질문(A)은 문장이 다르다. 단일 margin은 계수 w를 거쳐서만 공동 margin의 눈금으로 옮긴다.
  편차가 나와도 반례에 특이한 메커니즘의 증거는 아니다. 모델 내부의 계산에 대한 주장도 아니다.
  같은 이미지를 여러 pair가 돌려 쓰고, 일부 이미지는 두 클래스가 같이 쓴다. 구간은 클래스(20개)를 단위로 다시 뽑아 만든다."""


def main() -> None:
    parser = argparse.ArgumentParser(description="자리 가중치와 N 하락 몫")
    parser.add_argument("run", help="01_baseline과 02_decomposition/b_single_image 아래의 실행 폴더 이름")
    args = parser.parse_args()

    case_rows, _, _ = load_case_rows(args.run)
    rows = [row for row in with_single_margins(case_rows["A"], case_rows["B"])
            if row["variant"] in ["PPP", *ONE_N_VARIANTS]]
    pairs_by_class = class_pairs(rows)
    terms = weight_terms(pairs_by_class)
    estimates = weight_statistics(terms)

    nan_draws = {}  # 다시 뽑기마다 값을 정하지 못한(nan) 횟수를 statistic 안에서 적어 둔다

    def stacked(drawn: torch.Tensor) -> torch.Tensor:
        """뽑기마다의 통계량을 NAMES 순서로 쌓는다: (뽑기 수, 지표 수, 자리 3개)"""
        statistics = weight_statistics(drawn)
        values = torch.stack([statistics[name] for name in NAMES], dim=-2)
        nan_draws["count"] = values.isnan().sum(dim=0)
        return values

    intervals = class_bootstrap_interval(terms, stacked, LEVEL)  # (2, 지표 수, 자리 3개)

    def cell(name: str, slot: int) -> str:
        low, high = intervals[:, NAMES.index(name), slot].tolist()
        return interval_text(estimates[name][slot].item(), low, high)

    print(f"{'=' * 20} {args.run}: {RUN_ROLES.get(args.run, '역할을 정하지 않은 실행')} {'=' * 20}")
    print(f"pair {sum(len(pairs) for pairs in pairs_by_class.values())}개, 클래스 {len(pairs_by_class)}개. "
          f"자리 k는 N이 들어간 자리다: " + ", ".join(f"자리 {slot + 1} = {variant}"
                                            for slot, variant in enumerate(SLOT_VARIANTS)))
    header = "".join(f"{f'자리 {slot + 1} ({variant})':<30}" for slot, variant in enumerate(SLOT_VARIANTS))
    for title, group in GROUPS:
        print(f"\n[{title}]  값 [{LEVEL:.1%} 구간]")
        print(f"  {'':<22}{header}")
        for name, label in group:
            print(f"  {label:<22}" + "".join(f"{cell(name, slot):<32}" for slot in range(len(SLOT_VARIANTS))))

    gap = NAMES.index("share_gap")
    verdicts = [VERDICT[practical_verdict(*intervals[:, gap, slot].tolist(), EQUIVALENCE_DELTA)]
                for slot in range(len(SLOT_VARIANTS))]
    print(f"\n[판정]  r_k의 {LEVEL:.1%} 구간과 실질적 동등의 범위 ±{EQUIVALENCE_DELTA}")
    for slot, (variant, verdict) in enumerate(zip(SLOT_VARIANTS, verdicts)):
        print(f"  자리 {slot + 1} ({variant}): {cell('share_gap', slot)}  {verdict}")
    overall = ("가산 모형과 실질적 동등성 지지" if all(verdict == "실질적 동등" for verdict in verdicts)
               else "세 자리가 모두 실질적 동등은 아니다. 자리별 판정을 본다")
    print(f"  전체: {overall}")

    print(f"\n[다시 뽑기]  전체 {N_DRAWS}회. 지표마다 자리 1 / 2 / 3의 유효 횟수 (nan 횟수)")
    for name in NAMES:
        counts = nan_draws["count"][NAMES.index(name)].tolist()
        print(f"  {name:<18}" + " / ".join(f"{N_DRAWS - count} ({count})" for count in counts))
    print(f"\n{LIMITS}")


if __name__ == "__main__":
    main()
