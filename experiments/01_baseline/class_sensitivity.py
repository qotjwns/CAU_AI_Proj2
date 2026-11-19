"""
클래스 내부 민감도 분석: 같은 클래스 안에서 PPP의 한 자리를 N으로 바꾸면 답과 margin이 달라지는가
python experiments/01_baseline/class_sensitivity.py qwen2.5-vl-3b            # 주 분석
python experiments/01_baseline/class_sensitivity.py qwen2.5-vl-7b            # 모델 견고성 분석
python experiments/01_baseline/class_sensitivity.py qwen2.5-vl-3b_float32    # 정밀도 민감도 분석

저장된 A(01_baseline) 결과를 읽기만 한다. 모델은 쓰지 않는다. 표는 터미널에 출력하고,
클래스별 값은 outputs/tables/01_baseline/<실행 이름>_class_sensitivity.csv에도 저장한다.
값의 정의는 mivlm.analysis.class_sensitivity 참고. 그림은 plot_class_sensitivity.py가 그린다.

두 질문을 구분한다
    질문 1. 클래스 안에서 N 증거에 반응하는가: 같은 pair의 PPP와 N을 넣은 변형을 비교한다 (판정 R, Y, A)
    질문 2. False-Yes가 클래스의 Yes 성향을 따라가는가: 클래스 사이의 상관과 분산의 몫을 적기만 한다 (판정하지 않는다)
질문 1의 답이 "반응한다"여도 질문 2가 부정되지 않고, 그 반대도 마찬가지다.

판정 기준 (결과를 보기 전에 정했다. 결과를 본 뒤에 바꾸지 않는다)
    R. N을 넣으면 margin이 내려간다: 짝지은 margin 하락(세 변형 합침)의 클래스 평균.
       95% 구간 전체가 0보다 크면 지지, 전체가 0보다 작으면 반증
    Y. N을 넣으면 Yes 답이 줄어든다: Yes 비율 하락(세 변형 합침)의 클래스 평균. 기준은 R과 같다
    A. 클래스 내부 AUROC(세 변형 합침)의 클래스 평균: 95% 구간 전체가 0.45 ~ 0.55 안이면 "무차이 지지",
       하한이 0.55를 넘으면 "무차이 기각". 클래스 내부의 순위 구분력에 대한 판정일 뿐이고,
       "오류가 클래스 성향만으로 설명된다"의 증거로 읽지 않는다
    C. N의 자리에 따라 margin 하락이 다르다: 변형 둘씩의 차이 3개. 98.3% 구간(0.05를 3으로 나눔)이 0을 제외하면 지지
    그 밖에는 모두 미결정이다. 상관과 분산의 몫에는 판정 기준이 없다.
구간은 클래스를 단위로 10,000번 다시 뽑아 만든다 (seed 고정).

실행의 역할
    qwen2.5-vl-3b가 주 분석이다. qwen2.5-vl-7b는 다른 모델에서도 같은 판정이 나오는지(모델 견고성),
    qwen2.5-vl-3b_float32는 정밀도를 바꿔도 같은 판정이 나오는지(정밀도 민감도)를 본다.
    세 실행은 같은 케이스를 돌린 것이라 서로 독립된 표본이 아니다.
"""
import argparse
import csv

from mivlm.analysis.class_sensitivity import (
    ONE_N, RUN_ROLES, TRANSITIONS, class_pairs, class_values, difference_verdict, direction_verdict, explained_variance,
    leave_one_image_out, mean_interval, rank_correlation_interval, ranking_verdict, shared_images, without_images,
)
from mivlm.config import PREDICTIONS_DIR, TABLES_DIR
from mivlm.evaluation.metrics import ONE_N_VARIANTS, load_rows
from mivlm.evaluation.report import interval_text, print_class_table

EXPERIMENT = "01_baseline"
CONTRASTS = [("PPN", "PNP"), ("PPN", "NPP"), ("PNP", "NPP")]  # 자리끼리의 비교 세 쌍
CONTRAST_LEVEL = 1 - 0.05 / len(CONTRASTS)  # 세 쌍을 같이 보므로 구간을 넓힌다 (98.3%)
LOIO_KEYS = [f"margin_drop_{ONE_N}", f"yes_rate_drop_{ONE_N}", f"auroc_{ONE_N}"]  # 이미지 한 장을 뺀 범위를 적을 값
LIMITS = """[한계]
  독립된 사전 등록 실험이 아니다. 이 코드를 쓰기 전에 같은 데이터의 일부 값을 임시 계산으로 이미 봤다.
  구간은 클래스를 단위로 다시 뽑아 만든다. 단위가 적어(클래스 20개) 실제보다 조금 좁을 수 있다.
  클래스 하나의 값에는 구간이 없다. 이미지 8장을 여러 케이스가 돌려 써서 타당한 구간을 만들 수 없기 때문이다.
  세 실행(3B bf16, 7B bf16, 3B float32)은 같은 케이스를 돌린 것이라, 판정이 일치해도 독립된 재현이 아니다."""


def print_class_means(table: dict[str, dict], unshared: dict[str, dict]) -> None:
    """class_values의 값마다, 클래스 평균과 95% 구간을 전체 pair와 공유 이미지를 뺀 pair에서 나란히 출력한다"""
    print(f"\n[클래스 평균과 95% 구간]  왼쪽: 전체 pair (클래스 {len(table)}개) / "
          f"오른쪽: 두 클래스가 같이 쓰는 이미지가 든 pair 제외 (클래스 {len(unshared)}개)")
    for key in next(iter(table.values())):
        if key != "n_pairs":
            print(f"  {key:<32}{interval_text(*mean_interval([values[key] for values in table.values()])):<30}"
                  f"{interval_text(*mean_interval([values[key] for values in unshared.values()]))}")


def print_verdicts(table: dict[str, dict], unshared: dict[str, dict]) -> None:
    """판정 R, Y, A, C를 전체 pair와 공유 이미지를 뺀 pair에서 각각 출력한다 (기준은 파일 맨 위)"""
    print("\n[판정]  왼쪽: 전체 pair / 오른쪽: 공유 이미지가 든 pair 제외")
    claims = [
        ("R. N을 넣으면 margin이 내려간다", f"margin_drop_{ONE_N}", direction_verdict),
        ("Y. N을 넣으면 Yes 답이 줄어든다", f"yes_rate_drop_{ONE_N}", direction_verdict),
        ("A. 클래스 내부 AUROC (순위 구분력)", f"auroc_{ONE_N}", ranking_verdict),
    ]
    for label, key, verdict in claims:
        cells = []
        for values_by_class in [table, unshared]:
            mean, low, high = mean_interval([values[key] for values in values_by_class.values()])
            cells.append(f"{interval_text(mean, low, high)} {verdict(low, high)}")
        print(f"  {label:<32}{cells[0]:<44}{cells[1]}")
    for first, second in CONTRASTS:
        cells = []
        for values_by_class in [table, unshared]:
            mean, low, high = mean_interval([values[f"margin_drop_{first}"] - values[f"margin_drop_{second}"]
                                             for values in values_by_class.values()], CONTRAST_LEVEL)
            cells.append(f"{interval_text(mean, low, high)} {difference_verdict(low, high)}")
        print(f"  {f'C. margin 하락, {first} - {second} (98.3%)':<32}{cells[0]:<44}{cells[1]}")
    print("  A의 판정은 클래스 내부의 순위 구분력에 대한 것이다. 오류가 클래스 성향만으로 설명되는지는 판정하지 않는다.")


def print_class_tendency(rows: list[dict], table: dict[str, dict], pooled: dict[str, float]) -> None:
    """질문 2에 대한 기술통계: 클래스의 Yes 성향과 False-Yes, margin 하락 사이의 관계. 판정하지 않는다"""
    print("\n[질문 2의 기술통계: 클래스 성향과의 관계]  판정 기준 없음")
    for label, x_key, y_key in [
        ("PPP Yes 비율 대 N 1장 Yes 비율(False-Yes)", "yes_rate_PPP", f"yes_rate_{ONE_N}"),
        ("PPP 평균 margin 대 margin 하락", "mean_margin_PPP", f"margin_drop_{ONE_N}"),
    ]:
        correlation = rank_correlation_interval([values[x_key] for values in table.values()],
                                                [values[y_key] for values in table.values()])
        print(f"  클래스 사이의 순위 상관, {label}: {interval_text(*correlation)}")
    print("  위 첫 상관에는 같은 P 이미지가 PPP와 N 1장 케이스 양쪽에 들어가서 생기는 부분이 섞여 있다.")
    print(f"  margin의 분산 중 설명되는 몫: 클래스 {explained_variance(rows, ['object']):.2f}, "
          f"변형 {explained_variance(rows, ['variant']):.2f}, "
          f"클래스 x 변형 {explained_variance(rows, ['object', 'variant']):.2f}")
    auroc_mean = mean_interval([values[f"auroc_{ONE_N}"] for values in table.values()])[0]
    print(f"  AUROC(PPP 대 N 1장): 클래스를 섞어 한 번에 계산 {pooled[f'auroc_{ONE_N}']:.3f}, "
          f"클래스마다 계산한 값의 평균 {auroc_mean:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="클래스 내부 민감도 분석")
    parser.add_argument("run", help="outputs/predictions/01_baseline/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    rows = [row for row in load_rows(PREDICTIONS_DIR / EXPERIMENT / args.run)
            if row["variant"] in ["PPP", *ONE_N_VARIANTS]]
    pairs_by_class = class_pairs(rows)
    table = {name: class_values(pairs) for name, pairs in pairs_by_class.items()}
    table = dict(sorted(table.items(), key=lambda item: -item[1]["mean_margin_PPP"]))  # PPP margin이 높은 클래스부터
    all_pairs = [pair for pairs in pairs_by_class.values() for pair in pairs]
    shared = shared_images(pairs_by_class)
    unshared_pairs = {name: without_images(pairs, shared) for name, pairs in pairs_by_class.items()}
    unshared = {name: class_values(pairs) for name, pairs in unshared_pairs.items() if pairs}
    ranges = {name: leave_one_image_out(pairs_by_class[name], class_values) for name in table}

    print(f"{'=' * 20} {args.run}: {RUN_ROLES.get(args.run, '역할을 정하지 않은 실행')} {'=' * 20}")
    print(f"케이스 {len(rows)}개, 클래스 {len(table)}개, pair {len(all_pairs)}개")
    print(f"두 클래스가 같이 쓰는 이미지 {len(shared)}장. 그 이미지가 든 pair를 빼면 "
          f"pair {sum(len(pairs) for pairs in unshared_pairs.values())}개, 클래스 {len(unshared)}개가 남는다 "
          f"(클래스당 최소 {min(len(pairs) for pairs in unshared_pairs.values())}개)")

    print_class_table("클래스별 Yes 비율과 평균 margin (변형마다 케이스 수는 n)", table, {
        "n": ("n_pairs", "d"),
        **{f"yes {variant}": (f"yes_rate_{variant}", ".0%") for variant in ["PPP", *ONE_N_VARIANTS]},
        **{f"m {variant}": (f"mean_margin_{variant}", "+.2f") for variant in ["PPP", *ONE_N_VARIANTS]},
    })
    print_class_table("PPP 대비 변화: Yes 비율 하락(dYes), 짝지은 margin 하락(dm)", table, {
        **{f"dYes {name}": (f"yes_rate_drop_{name}", "+.0%") for name in [*ONE_N_VARIANTS, ONE_N]},
        **{f"dm {name}": (f"margin_drop_{name}", "+.2f") for name in [*ONE_N_VARIANTS, ONE_N]},
    })
    print_class_table("margin 하락이 0보다 큰 pair의 비율(dm>0), 클래스 내부 AUROC", table, {
        **{f"dm>0 {name}": (f"margin_drop_positive_rate_{name}", ".0%") for name in [*ONE_N_VARIANTS, ONE_N]},
        **{f"AUC {name}": (f"auroc_{name}", ".3f") for name in [*ONE_N_VARIANTS, ONE_N]},
    })
    print_class_table("답의 전환, 세 변형 합침 (PPP에서의 답 -> N을 넣은 변형에서의 답)", table, {
        "yes PPP": ("yes_rate_PPP", ".0%"),
        **{transition: (f"{transition}_{ONE_N}", ".0%") for transition in TRANSITIONS},
    })

    print("\n[이미지 한 장을 뺐을 때의 범위]  값 [최소, 최대]. 구간이 아니라 이미지 한 장에 좌우되는 정도다")
    print(f"{'':<14}{'min pairs':>10}" + "".join(f"{key:>30}" for key in LOIO_KEYS))
    for name, values in table.items():
        cells = [f"{values[key]:+.2f} [{ranges[name][key][0]:+.2f}, {ranges[name][key][1]:+.2f}]" for key in LOIO_KEYS]
        print(f"{name:<14}{ranges[name]['n_pairs'][0]:>10d}" + "".join(f"{cell:>30}" for cell in cells))

    print_class_means(table, unshared)
    print_verdicts(table, unshared)
    print_class_tendency(rows, table, class_values(all_pairs))
    print(f"\n{LIMITS}")

    table_path = TABLES_DIR / EXPERIMENT / f"{args.run}_class_sensitivity.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        keys = list(next(iter(table.values())))
        writer.writerow(["object", *keys, *[f"{key}_{end}" for key in LOIO_KEYS for end in ["loio_min", "loio_max"]]])
        for name, values in table.items():
            writer.writerow([name, *[values[key] for key in keys],
                             *[end for key in LOIO_KEYS for end in ranges[name][key]]])
    print(f"\n[ok] {table_path}")


if __name__ == "__main__":
    main()
