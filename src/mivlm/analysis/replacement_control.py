"""
P→P 대조군: PPP의 한 자리를 다른 P 이미지로 바꿨을 때와 N 이미지로 바꿨을 때를, 같은 출발 케이스에서 비교한다

입력은 A(01_baseline) 실행의 행이다 (mivlm.evaluation.metrics.load_rows).

묶음: 한 그룹의 PPP 두 개가 한 자리만 다를 때, 그 둘(A, B)과 각자의 그 자리를 N으로 바꾼 케이스(A_N, B_N).
    네 케이스는 나머지 두 자리의 이미지가 같다. 예: 자리 3의 묶음
        A = PPP [a, b, c]    B = PPP [a, b, d]    A_N = PPN [a, b, n1]    B_N = PPN [a, b, n2]
출발: 묶음마다 A에서 출발하는 경우와 B에서 출발하는 경우를 둘 다 쓴다. 방향을 고르지 않는다.
    A에서 출발하면 P→P는 A -> B, P→N은 A -> A_N이다. 같은 출발, 같은 자리, 같은 나머지 두 이미지다
    변화 = margin(출발) - margin(바꾼 케이스). 양수면 No 쪽으로 움직인 것이다
    핵심 대비 = (P→N의 변화) - (P→P의 변화) = margin(다른 P로 바꾼 케이스) - margin(N으로 바꾼 케이스)

두 방향을 다 쓰기 때문에 결과와 상관없이 정해지는 것
    P→P 변화의 평균은 0이다. 그래서 핵심 대비의 평균은 같은 출발들의 P→N 변화의 평균과 같다
    P→P에서 변화가 양수인 비율은 (1 - 변화가 0인 비율) / 2이고, Yes->No의 비율과 No->Yes의 비율은 같다
한 묶음의 두 출발은 같은 네 케이스로 만든 것이라 서로 독립이 아니다. 출발을 독립된 표본으로 세지 않고,
구간은 클래스를 단위로 만든다 (mivlm.analysis.class_sensitivity.mean_interval).

묶음은 pair와 같은 모양({"pair", "cases"})이라, class_sensitivity의 pair_images, shared_images, without_images,
leave_one_image_out을 그대로 쓸 수 있다.
"""
import statistics
from collections import Counter

import torch

from mivlm.analysis.class_sensitivity import TRANSITIONS, ranking_verdict
from mivlm.data.cases import case_pairs
from mivlm.evaluation.metrics import ONE_N_VARIANTS

ABS_QUANTILES = {"median": 0.5, "q75": 0.75, "q90": 0.9, "q95": 0.95}  # P→P 변화의 크기(절댓값)에서 적을 분위수


def replacement_quads(rows: list[dict]) -> dict[str, list[dict]]:
    """행에서 묶음을 만든다: {클래스: [묶음, ...]}

    묶음: {"pair": 이름, "slot": 바뀌는 자리(1, 2, 3), "cases": {"A": 행, "B": 행, "A_N": 행, "B_N": 행}}
    같은 그룹의 PPP 둘이 정확히 한 자리에서만 다르면 묶음 하나가 된다. A가 먼저 나온 pair의 PPP다.
    A_N, B_N은 각 pair에서 그 자리에 N이 들어간 변형이다 (자리 1은 NPP, 자리 2는 PNP, 자리 3은 PPN).
    네 케이스의 나머지 두 자리 이미지가 같지 않으면 에러를 낸다.
    """
    pairs_by_group = {}
    for pair in case_pairs([row for row in rows if row["variant"] in ["PPP", *ONE_N_VARIANTS]]):
        pairs_by_group.setdefault(pair["cases"]["PPP"]["group_id"], []).append(pair["cases"])

    quads_by_class = {}
    for group_id, group in pairs_by_group.items():
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                differing = [slot for slot in range(3)
                             if first["PPP"]["image_ids"][slot] != second["PPP"]["image_ids"][slot]]
                if len(differing) != 1:
                    continue
                slot = differing[0]
                variant = "".join("N" if position == slot else "P" for position in range(3))
                cases = {"A": first["PPP"], "B": second["PPP"], "A_N": first[variant], "B_N": second[variant]}
                for case in cases.values():
                    if any(case["image_ids"][position] != cases["A"]["image_ids"][position]
                           for position in range(3) if position != slot):
                        raise ValueError(f"{group_id}의 자리 {slot + 1} 묶음: {case['case_id']}의 나머지 두 자리가 A와 다름")
                quads_by_class.setdefault(first["PPP"]["object"], []).append(
                    {"pair": f"{group_id}_slot{slot + 1}", "slot": slot + 1, "cases": cases})
    return quads_by_class


def same_n_image(quad: dict) -> bool:
    """묶음의 두 N 교체 케이스가 같은 N 이미지를 쓰는가. 그러면 A_N과 B_N은 이미지 세 장이 모두 같은 입력이다"""
    return quad["cases"]["A_N"]["image_ids"] == quad["cases"]["B_N"]["image_ids"]


def quad_starts(quad: dict) -> list[dict]:
    """묶음의 출발 두 개 (A에서 출발, B에서 출발)

    출발마다: start_yes (출발의 답이 Yes인가), pp_change와 pn_change (P→P, P→N의 margin 변화),
    pp_transition과 pn_transition ("출발의 답->바꾼 케이스의 답")
    """
    cases = quad["cases"]
    starts = []
    for start, other_p, with_n in [("A", "B", "A_N"), ("B", "A", "B_N")]:
        before = cases[start]
        starts.append({
            "start_yes": before["prediction"] == "Yes",
            "pp_change": before["logit_margin"] - cases[other_p]["logit_margin"],
            "pn_change": before["logit_margin"] - cases[with_n]["logit_margin"],
            "pp_transition": f"{before['prediction']}->{cases[other_p]['prediction']}",
            "pn_transition": f"{before['prediction']}->{cases[with_n]['prediction']}",
        })
    return starts


def control_values(quads: list[dict]) -> dict[str, float]:
    """묶음들(보통 한 클래스의 묶음 전부)의 출발 전체로 계산한 값. pp는 P→P, pn은 P→N이다

    n_starts                        출발 수 (묶음 수의 2배). 독립된 표본의 수가 아니다
    pn_change_mean, _median         P→N 변화의 평균, 중앙값
    pn_drop_rate, pp_drop_rate      변화가 0보다 큰 출발의 비율
    pp_change_mean, pp_tie_rate     P→P 변화의 평균(항상 0), 변화가 0인 출발의 비율
    pp_abs_median, _q75, _q90, _q95 P→P 변화의 크기(절댓값)의 분위수. "P끼리 바꿀 때 margin이 흔들리는 정도"다
    contrast_mean                   핵심 대비의 평균 (S1). pn_change_mean과 같은 값이다
    contrast_positive_rate          핵심 대비가 0보다 큰 출발의 비율, 0인 출발은 반으로 센다 (S2)
    pn_minus_abs_pp_mean            P→N 변화 - P→P 변화의 크기의 평균 (M)
    pp_Yes->No 등, pn_Yes->No 등    전환 비율 (전체 출발 중)
    yes_to_no_difference            pn_Yes->No - pp_Yes->No (T)
    n_start_yes                     출발의 답이 Yes인 출발의 수. 아래 조건부 비율의 분모다
    pp_yes_to_no_given_yes 등       출발이 Yes인 경우만 놓고 본 Yes->No 비율과 그 차이. n_start_yes가 0이면 nan
    """
    starts = [start for quad in quads for start in quad_starts(quad)]
    pp = [start["pp_change"] for start in starts]
    pn = [start["pn_change"] for start in starts]
    contrasts = [after_n - after_p for after_n, after_p in zip(pn, pp)]
    pp_abs = torch.tensor([abs(change) for change in pp], dtype=torch.float64)
    pp_transitions = Counter(start["pp_transition"] for start in starts)
    pn_transitions = Counter(start["pn_transition"] for start in starts)
    n_start_yes = sum(start["start_yes"] for start in starts)
    pp_given_yes = pp_transitions["Yes->No"] / n_start_yes if n_start_yes else float("nan")
    pn_given_yes = pn_transitions["Yes->No"] / n_start_yes if n_start_yes else float("nan")
    return {
        "n_starts": len(starts),
        "pn_change_mean": statistics.mean(pn),
        "pn_change_median": statistics.median(pn),
        "pn_drop_rate": sum(change > 0 for change in pn) / len(starts),
        "pp_change_mean": statistics.mean(pp),
        "pp_drop_rate": sum(change > 0 for change in pp) / len(starts),
        "pp_tie_rate": sum(change == 0 for change in pp) / len(starts),
        **{f"pp_abs_{name}": pp_abs.quantile(q).item() for name, q in ABS_QUANTILES.items()},
        "contrast_mean": statistics.mean(contrasts),
        "contrast_positive_rate": sum(1 if c > 0 else 0.5 if c == 0 else 0 for c in contrasts) / len(starts),
        "pn_minus_abs_pp_mean": statistics.mean(after_n - abs(after_p) for after_n, after_p in zip(pn, pp)),
        **{f"pp_{transition}": pp_transitions[transition] / len(starts) for transition in TRANSITIONS},
        **{f"pn_{transition}": pn_transitions[transition] / len(starts) for transition in TRANSITIONS},
        "yes_to_no_difference": (pn_transitions["Yes->No"] - pp_transitions["Yes->No"]) / len(starts),
        "n_start_yes": n_start_yes,
        "pp_yes_to_no_given_yes": pp_given_yes,
        "pn_yes_to_no_given_yes": pn_given_yes,
        "yes_to_no_given_yes_difference": pn_given_yes - pp_given_yes,
    }


def dominance_verdict(low: float, high: float) -> str:
    """S2의 판정: "N으로 바꾼 쪽이 다른 P로 바꾼 쪽보다 margin이 낮은 경우가 더 많다"

    통계량은 핵심 대비가 0보다 큰 출발의 비율이고, 0.5면 두 교체가 구분되지 않는 것이다.
    구간 하한이 0.55를 넘으면 지지, 구간 전체가 0.45 ~ 0.55 안이면 반증(구분되지 않음), 그 밖에는 미결정.
    0.55는 연구자가 정한 실질적 효과의 기준이다 (mivlm.analysis.class_sensitivity.NO_DIFFERENCE와 같은 범위).
    """
    return {"무차이 기각": "지지", "무차이 지지": "반증", "미결정": "미결정"}[ranking_verdict(low, high)]
