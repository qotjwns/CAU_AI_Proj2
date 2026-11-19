"""
P→P 대조군 분석: PPP의 한 자리를 N으로 바꿀 때의 margin 하락이, 다른 P로 바꿀 때의 변화와 구분되는가
python experiments/01_baseline/replacement_control.py qwen2.5-vl-3b            # 주 분석
python experiments/01_baseline/replacement_control.py qwen2.5-vl-7b            # 보조 분석: 모델 견고성
python experiments/01_baseline/replacement_control.py qwen2.5-vl-3b_float32    # 보조 분석: 정밀도 민감도

저장된 A(01_baseline) 결과를 읽기만 한다. 모델은 쓰지 않는다. 표는 터미널에 출력하고,
클래스별 값은 outputs/tables/01_baseline/<실행 이름>_replacement_control.csv에도 저장한다.
묶음, 출발, 핵심 대비의 정의는 mivlm.analysis.replacement_control 참고.

짝짓기: 한 그룹의 PPP 둘이 한 자리만 다르면 묶음 하나다. 출발 하나에 대해 P→P(같은 자리를 다른 P로)와
    P→N(같은 자리를 N으로)을 비교한다. 클래스, 자리, 출발 케이스, 나머지 두 이미지가 같다.
방향: 묶음마다 두 방향(A에서 출발, B에서 출발)을 모두 쓴다. 고르지 않으므로 방향 때문에 생기는 치우침이 없다.
    한 묶음의 두 출발은 같은 네 케이스로 만든 것이라 독립이 아니다. 출발 수를 표본 수로 쓰지 않고, 클래스를 단위로 본다.

판정 기준 (P→P의 값을 보기 전에 정했다. 결과를 본 뒤에 바꾸지 않는다)
    S2 (주 판정). 핵심 대비가 0보다 큰 출발의 비율의 클래스 평균. 0.5면 N 교체와 P 교체가 구분되지 않는 것이다.
        95% 구간 하한이 0.55를 넘으면 지지, 구간 전체가 0.45 ~ 0.55 안이면 반증. 0.55는 연구자가 정한 실질적 효과의 기준이다
        자리마다 따로 볼 때는 세 자리를 같이 보므로 98.3% 구간을 쓴다
    M (보조). P→N 변화 - P→P 변화의 크기의 클래스 평균. 95% 구간 전체가 0보다 크면 지지, 전체가 0보다 작으면 반증
    T (보조). Yes->No 비율(P→N) - Yes->No 비율(P→P)의 클래스 평균. 기준은 M과 같다.
        전체 출발 기준과, 출발의 답이 Yes인 경우만 놓고 본 조건부 기준을 둘 다 낸다
    S1. 핵심 대비의 클래스 평균. 기준은 M과 같다. 두 방향을 다 쓰면 P→P 변화의 평균이 0이라, 이 값은 같은 출발들의
        P→N 변화의 평균과 같다. 기존 P→N 분석의 부분집합을 다시 적은 것이고, 새로운 근거가 아니다
    그 밖에는 모두 미결정이다.
구간은 클래스를 단위로 10,000번 다시 뽑아 만든다 (seed 고정).

민감도 분석: 두 클래스가 같이 쓰는 이미지가 든 묶음 제외 / 두 N 교체 케이스가 같은 N 이미지를 쓰는 묶음 제외 /
    클래스마다 이미지 한 장을 뺐을 때의 범위. 모두 묶음을 통째로 빼서 짝을 유지한다.
"""
import argparse
import csv
import math

from mivlm.analysis.class_sensitivity import (
    RUN_ROLES, TRANSITIONS, direction_verdict, leave_one_image_out, mean_interval, shared_images, without_images,
)
from mivlm.analysis.replacement_control import (
    ABS_QUANTILES, control_values, dominance_verdict, replacement_quads, same_n_image,
)
from mivlm.config import PREDICTIONS_DIR, TABLES_DIR
from mivlm.evaluation.metrics import load_rows
from mivlm.evaluation.report import interval_text, print_class_table

EXPERIMENT = "01_baseline"
SLOTS = [1, 2, 3]
SLOT_LEVEL = 1 - 0.05 / len(SLOTS)  # 세 자리를 같이 보므로 구간을 넓힌다 (98.3%)
LOIO_KEYS = ["contrast_positive_rate", "pn_minus_abs_pp_mean"]  # 이미지 한 장을 뺀 범위를 적을 값 (S2, M)
SOURCES = ["전체 묶음", "공유 이미지 제외", "같은 N 이미지 제외"]  # 판정을 나란히 적는 세 가지 묶음 집합
LIMITS = """[한계]
  탐색적 대조 분석이다. 기존 P→N 결과를 본 뒤에 설계했고, 독립된 확증이 아니다.
  P→P는 완전한 인과적 대조군이 아니다. N 이미지는 객체가 없다는 것 말고도 P 이미지와 다르다(장면 구성, 큰 객체의 유무).
  여기서 보이는 것은 "N 종류의 이미지로 바꿀 때와 P 종류의 이미지로 바꿀 때의 차이"까지다.
  N이 1장인 케이스의 절반만 짝이 있고, 그룹마다 pair 2와 3은 두 자리에서 두 번 쓰인다.
  같은 이미지를 여러 케이스가 돌려 쓰고, 일부 이미지는 두 클래스가 같이 쓴다. 출발은 독립된 표본이 아니다.
  구간은 클래스를 단위로 다시 뽑아 만든다. 단위가 적어(클래스 20개) 실제보다 조금 좁을 수 있다.
  세 실행(3B bf16, 7B bf16, 3B float32)은 같은 케이스를 돌린 것이라, 판정이 일치해도 독립된 재현이 아니다."""


def class_tables(quads_by_class: dict[str, list[dict]]) -> dict[str, dict]:
    """클래스마다의 control_values. 묶음이 남지 않은 클래스는 뺀다"""
    return {name: control_values(quads) for name, quads in quads_by_class.items() if quads}


def only_slot(quads_by_class: dict[str, list[dict]], slot: int) -> dict[str, list[dict]]:
    """클래스마다 자리 slot의 묶음만 남긴다"""
    return {name: [quad for quad in quads if quad["slot"] == slot] for name, quads in quads_by_class.items()}


def verdict_cells(tables: list[dict[str, dict]], key: str, verdict, level: float = 0.95) -> str:
    """tables의 표마다: key의 클래스 평균과 구간, 판정, 쓴 클래스 수. 값을 정할 수 없는 클래스(nan)는 뺀다"""
    cells = []
    for table in tables:
        per_class = [values[key] for values in table.values() if not math.isnan(values[key])]
        mean, low, high = mean_interval(per_class, level)
        cells.append(f"{interval_text(mean, low, high)} {verdict(low, high)} ({len(per_class)}개)")
    return "".join(f"{cell:<40}" for cell in cells)


def main() -> None:
    parser = argparse.ArgumentParser(description="P→P 대조군 분석")
    parser.add_argument("run", help="outputs/predictions/01_baseline/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    quads_by_class = replacement_quads(load_rows(PREDICTIONS_DIR / EXPERIMENT / args.run))
    all_quads = [quad for quads in quads_by_class.values() for quad in quads]
    shared = shared_images(quads_by_class)
    # 판정을 나란히 볼 세 가지 묶음 집합. 순서는 SOURCES와 같다
    sources = [
        quads_by_class,
        {name: without_images(quads, shared) for name, quads in quads_by_class.items()},
        {name: [quad for quad in quads if not same_n_image(quad)] for name, quads in quads_by_class.items()},
    ]
    tables = [class_tables(source) for source in sources]
    table = tables[0]
    ranges = {name: leave_one_image_out(quads_by_class[name], control_values) for name in table}

    print(f"{'=' * 20} {args.run}: {RUN_ROLES.get(args.run, '역할을 정하지 않은 실행')} {'=' * 20}")
    print(f"묶음 {len(all_quads)}개, 출발 {2 * len(all_quads)}개 (독립된 표본의 수가 아니다), 클래스 {len(table)}개")
    for name, source in zip(SOURCES[1:], sources[1:]):
        print(f"{name}: 묶음 {sum(len(quads) for quads in source.values())}개, "
              f"클래스 {sum(bool(quads) for quads in source.values())}개")

    print_class_table("클래스별: P→N 변화, P→P 변화의 크기, 핵심 대비 (S1 = 평균, S2 = 양수인 비율), M", table, {
        "starts": ("n_starts", "d"),
        "pn mean": ("pn_change_mean", "+.2f"),
        **{f"|pp| {name}": (f"pp_abs_{name}", ".2f") for name in ABS_QUANTILES},
        "S1 mean": ("contrast_mean", "+.2f"),
        "S2 share": ("contrast_positive_rate", ".0%"),
        "M": ("pn_minus_abs_pp_mean", "+.2f"),
    })
    print_class_table("클래스별: Yes->No 전환. 전체 출발 중의 비율(pp, pn)과 출발이 Yes인 경우만의 비율(|Yes)", table, {
        "starts": ("n_starts", "d"),
        "pp Y->N": ("pp_Yes->No", ".0%"),
        "pn Y->N": ("pn_Yes->No", ".0%"),
        "n Yes": ("n_start_yes", "d"),
        "pp |Yes": ("pp_yes_to_no_given_yes", ".0%"),
        "pn |Yes": ("pn_yes_to_no_given_yes", ".0%"),
    })

    print("\n[P→P 변화의 크기(절댓값)의 분포]  클래스를 섞어 묶음 전체에서 계산. 옆의 pn은 같은 출발들의 P→N 변화")
    print(f"{'':<10}{'quads':>8}" + "".join(f"{name:>9}" for name in ABS_QUANTILES)
          + f"{'pn median':>12}{'pn mean':>10}")
    by_slot = {f"slot {slot}": [quad for quads in only_slot(quads_by_class, slot).values() for quad in quads]
               for slot in SLOTS}
    for label, quads in {"all": all_quads, **by_slot}.items():
        pooled = control_values(quads)
        print(f"{label:<10}{len(quads):>8}" + "".join(f"{pooled[f'pp_abs_{name}']:>9.2f}" for name in ABS_QUANTILES)
              + f"{pooled['pn_change_median']:>12.2f}{pooled['pn_change_mean']:>10.2f}")

    pooled = control_values(all_quads)
    print("\n[두 방향을 다 써서 정해지는 값]  결과가 아니라 확인용이다")
    print(f"  P→P 변화의 평균 {pooled['pp_change_mean']:+.3f} (0이어야 한다), "
          f"양수인 비율 {pooled['pp_drop_rate']:.3f} = (1 - 0인 비율 {pooled['pp_tie_rate']:.3f}) / 2")
    print(f"  P→P 전환: Yes->No {pooled['pp_Yes->No']:.3f}, No->Yes {pooled['pp_No->Yes']:.3f} (같아야 한다)")
    print("  P→P 전환 전체: " + ", ".join(f"{t} {pooled[f'pp_{t}']:.3f}" for t in TRANSITIONS))
    print("  P→N 전환 전체: " + ", ".join(f"{t} {pooled[f'pn_{t}']:.3f}" for t in TRANSITIONS))

    print(f"\n[판정]  클래스 평균 [구간] 판정 (쓴 클래스 수). 열: {' / '.join(SOURCES)}")
    print(f"  {'S2 (주 판정). 핵심 대비가 양수인 비율':<36}{verdict_cells(tables, 'contrast_positive_rate', dominance_verdict)}")
    for slot in SLOTS:
        slot_tables = [class_tables(only_slot(source, slot)) for source in sources]
        print(f"  {f'S2, 자리 {slot}만 (98.3% 구간)':<36}"
              f"{verdict_cells(slot_tables, 'contrast_positive_rate', dominance_verdict, SLOT_LEVEL)}")
    print(f"  {'M (보조). P→N 변화 - |P→P 변화|':<36}{verdict_cells(tables, 'pn_minus_abs_pp_mean', direction_verdict)}")
    print(f"  {'T (보조). Yes->No 비율의 차이, 전체 출발':<36}{verdict_cells(tables, 'yes_to_no_difference', direction_verdict)}")
    print(f"  {'T (보조). 출발이 Yes인 경우만':<36}"
          f"{verdict_cells(tables, 'yes_to_no_given_yes_difference', direction_verdict)}")
    print(f"  {'S1. 핵심 대비의 평균':<36}{verdict_cells(tables, 'contrast_mean', direction_verdict)}")
    print("  S2의 0.55는 연구자가 정한 실질적 효과의 기준이다. S1은 기존 P→N 분석의 부분집합을 다시 적은 것이다.")
    print("  조건부 T는 출발이 Yes인 경우가 없는 클래스를 뺀다. 클래스별 분모는 위 표의 n Yes다.")

    print("\n[이미지 한 장을 뺐을 때의 범위]  값 [최소, 최대]. 구간이 아니라 이미지 한 장에 좌우되는 정도다")
    print(f"{'':<14}{'min starts':>11}" + "".join(f"{key:>32}" for key in LOIO_KEYS))
    for name, values in table.items():
        cells = [f"{values[key]:+.2f} [{ranges[name][key][0]:+.2f}, {ranges[name][key][1]:+.2f}]" for key in LOIO_KEYS]
        print(f"{name:<14}{ranges[name]['n_starts'][0]:>11d}" + "".join(f"{cell:>32}" for cell in cells))

    print(f"\n{LIMITS}")

    table_path = TABLES_DIR / EXPERIMENT / f"{args.run}_replacement_control.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    slot_shares = {slot: class_tables(only_slot(quads_by_class, slot)) for slot in SLOTS}
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        keys = list(next(iter(table.values())))
        writer.writerow(["object", *keys, *[f"contrast_positive_rate_slot{slot}" for slot in SLOTS],
                         *[f"{key}_{end}" for key in LOIO_KEYS for end in ["loio_min", "loio_max"]]])
        for name, values in table.items():
            writer.writerow([name, *[values[key] for key in keys],
                             *[slot_shares[slot][name]["contrast_positive_rate"] for slot in SLOTS],
                             *[end for key in LOIO_KEYS for end in ranges[name][key]]])
    print(f"\n[ok] {table_path}")


if __name__ == "__main__":
    main()
