"""
src/mivlm/analysis/replacement_control.py 단위 테스트 (모델도, 저장된 결과도 쓰지 않는다)
pytest tests/unit/test_replacement_control.py -v
"""
import math

import pytest

from mivlm.analysis.class_sensitivity import leave_one_image_out, pair_images, shared_images, without_images
from mivlm.analysis.replacement_control import (
    control_values, dominance_verdict, quad_starts, replacement_quads, same_n_image,
)

# 그룹 하나의 P 이미지 1, 2, 3, 4로 만든 pair 네 개 (실제 케이스 파일과 같은 구성)
P_IMAGES = {1: [1, 2, 3], 2: [1, 2, 4], 3: [1, 3, 4], 4: [2, 3, 4]}
N_IMAGE = {1: 91, 2: 92, 3: 93, 4: 93}  # pair 3과 4는 같은 N 이미지를 쓴다
# pair마다 PPP, PPN, PNP, NPP의 margin. 답은 margin의 부호로 정한다 (양수면 Yes)
MARGINS = {
    1: {"PPP": 2.0, "PPN": -1.0, "PNP": 0.5, "NPP": 1.0},
    2: {"PPP": 1.0, "PPN": -2.0, "PNP": -0.5, "NPP": 0.5},
    3: {"PPP": -0.5, "PPN": -1.5, "PNP": -1.0, "NPP": -2.0},
    4: {"PPP": 0.5, "PPN": -0.5, "PNP": 0.25, "NPP": -2.0},
}


def _rows(name: str = "cat", margins: dict = MARGINS) -> list[dict]:
    """한 클래스의 그룹 하나: pair 4개 x 변형 4개의 행"""
    rows = []
    for pair_id, p in P_IMAGES.items():
        n = N_IMAGE[pair_id]
        images = {"PPP": p, "PPN": [p[0], p[1], n], "PNP": [p[0], n, p[2]], "NPP": [n, p[1], p[2]]}
        for variant, image_ids in images.items():
            margin = margins[pair_id][variant]
            rows.append({"case_id": f"group_01_{name}_{pair_id}_{variant}", "group_id": f"group_01_{name}",
                         "pair_id": pair_id, "object": name, "variant": variant, "image_ids": image_ids,
                         "label": "Yes" if variant == "PPP" else "No", "prediction": "Yes" if margin > 0 else "No",
                         "logit_margin": margin})
    return rows


# ----------------------------------------------------------- replacement_quads


def test_each_pair_of_ppp_cases_that_differ_in_one_slot_becomes_a_quad():
    quads = replacement_quads(_rows())["cat"]

    # pair (1, 2)는 자리 3만, (2, 3)은 자리 2만, (3, 4)는 자리 1만 다르다. 나머지 조합은 두 자리 이상 다르다
    assert [(quad["pair"], quad["slot"]) for quad in quads] == [
        ("group_01_cat_slot3", 3), ("group_01_cat_slot2", 2), ("group_01_cat_slot1", 1)]
    slot_3 = quads[0]["cases"]
    assert {name: case["case_id"] for name, case in slot_3.items()} == {
        "A": "group_01_cat_1_PPP", "B": "group_01_cat_2_PPP", "A_N": "group_01_cat_1_PPN", "B_N": "group_01_cat_2_PPN"}
    slot_1 = quads[2]["cases"]
    slot_1_images = [slot_1[name]["image_ids"] for name in ["A", "B", "A_N", "B_N"]]
    assert slot_1_images == [[1, 3, 4], [2, 3, 4], [93, 3, 4], [93, 3, 4]]


def test_quads_are_grouped_by_class():
    quads_by_class = replacement_quads(_rows("cat") + _rows("dog"))

    assert list(quads_by_class) == ["cat", "dog"]
    assert [len(quads) for quads in quads_by_class.values()] == [3, 3]


def test_a_quad_whose_cases_do_not_share_the_other_two_images_is_an_error():
    rows = _rows()
    next(row for row in rows if row["case_id"] == "group_01_cat_2_PPN")["image_ids"] = [1, 9, 92]  # 자리 2가 달라졌다

    with pytest.raises(ValueError, match="group_01_cat_2_PPN"):
        replacement_quads(rows)


def test_same_n_image_tells_whether_the_two_n_replaced_cases_are_the_same_input():
    quads = replacement_quads(_rows())["cat"]

    assert [same_n_image(quad) for quad in quads] == [False, False, True]  # 자리 1의 묶음만 pair 3, 4의 같은 N을 쓴다


def test_quads_work_with_the_image_functions_written_for_pairs():
    quads = replacement_quads(_rows())["cat"]

    assert pair_images(quads[0]) == {1, 2, 3, 4, 91, 92}
    assert [quad["slot"] for quad in without_images(quads, {91})] == [2, 1]  # 이미지 91은 자리 3의 묶음에만 있다
    assert shared_images(replacement_quads(_rows("cat") + _rows("dog"))) == {1, 2, 3, 4, 91, 92, 93}


# ----------------------------------------------------------------- quad_starts


def test_quad_starts_use_both_directions_from_the_same_four_cases():
    slot_3 = replacement_quads(_rows())["cat"][0]  # A +2.0, B +1.0, A_N -1.0, B_N -2.0

    from_a, from_b = quad_starts(slot_3)

    assert from_a == {"start_yes": True, "pp_change": 1.0, "pn_change": 3.0,
                      "pp_transition": "Yes->Yes", "pn_transition": "Yes->No"}
    assert from_b == {"start_yes": True, "pp_change": -1.0, "pn_change": 3.0,
                      "pp_transition": "Yes->Yes", "pn_transition": "Yes->No"}
    assert from_a["pp_change"] == -from_b["pp_change"]  # P→P는 방향을 뒤집으면 부호만 바뀐다


# -------------------------------------------------------------- control_values


def test_control_values_are_hand_computed_over_the_six_starts():
    # 출발 6개 (묶음 순서대로, A에서 출발과 B에서 출발)
    #   자리 3: P→P +1.0, -1.0 / P→N 3.0, 3.0 / 대비 2.0, 4.0
    #   자리 2: P→P +1.5, -1.5 / P→N 1.5, 0.5 / 대비 0.0, 2.0   (A +1.0, B -0.5, A_N -0.5, B_N -1.0)
    #   자리 1: P→P -1.0, +1.0 / P→N 1.5, 2.5 / 대비 2.5, 1.5   (A -0.5, B +0.5, A_N -2.0, B_N -2.0)
    values = control_values(replacement_quads(_rows())["cat"])

    assert values["n_starts"] == 6
    assert values["pn_change_mean"] == 2.0  # (3.0 + 3.0 + 1.5 + 0.5 + 1.5 + 2.5) / 6
    assert values["pn_change_median"] == 2.0  # 정렬하면 0.5, 1.5, 1.5, 2.5, 3.0, 3.0
    assert values["pn_drop_rate"] == 1.0
    assert values["contrast_mean"] == 2.0  # (2.0 + 4.0 + 0.0 + 2.0 + 2.5 + 1.5) / 6
    assert values["contrast_positive_rate"] == pytest.approx(5.5 / 6)  # 양수 5개, 0인 것 1개는 반으로 센다
    # P→N - |P→P|: 2.0, 2.0, 0.0, -1.0, 0.5, 1.5
    assert values["pn_minus_abs_pp_mean"] == pytest.approx(5.0 / 6)


def test_using_both_directions_fixes_some_p_to_p_values_whatever_the_margins_are():
    values = control_values(replacement_quads(_rows())["cat"])

    assert values["pp_change_mean"] == 0.0
    assert values["pp_tie_rate"] == 0.0
    assert values["pp_drop_rate"] == 0.5  # (1 - 0인 비율) / 2
    assert values["pp_Yes->No"] == values["pp_No->Yes"]
    assert values["contrast_mean"] == values["pn_change_mean"]  # S1은 P→N 변화의 평균과 같은 값이다


def test_absolute_p_to_p_changes_are_reported_as_quantiles():
    values = control_values(replacement_quads(_rows())["cat"])

    # |P→P| 여섯 개를 정렬하면 1.0, 1.0, 1.0, 1.0, 1.5, 1.5
    assert values["pp_abs_median"] == 1.0
    assert values["pp_abs_q75"] == pytest.approx(1.375)  # 0.75 x 5 = 3.75번째: 1.0과 1.5 사이의 0.75 지점
    assert values["pp_abs_q90"] == pytest.approx(1.5)
    assert values["pp_abs_q95"] == pytest.approx(1.5)


def test_transitions_are_given_over_all_starts_and_over_the_starts_answered_yes():
    values = control_values(replacement_quads(_rows())["cat"])

    # P→P: 자리 3은 Yes->Yes 둘, 자리 2는 Yes->No와 No->Yes, 자리 1은 No->Yes와 Yes->No
    order = ["Yes->No", "Yes->Yes", "No->No", "No->Yes"]
    assert [values[f"pp_{t}"] for t in order] == pytest.approx([2 / 6, 2 / 6, 0, 2 / 6])
    # P→N: Yes에서 출발한 4개는 모두 No로, No에서 출발한 2개는 No로 남는다
    assert [values[f"pn_{t}"] for t in order] == pytest.approx([4 / 6, 0, 2 / 6, 0])
    assert values["yes_to_no_difference"] == pytest.approx(2 / 6)
    # 출발이 Yes인 경우는 4개 (pair 1, 2, 2, 4의 PPP). 그중 P→P로 No가 된 것 2개, P→N으로 No가 된 것 4개
    assert values["n_start_yes"] == 4
    assert values["pp_yes_to_no_given_yes"] == 0.5
    assert values["pn_yes_to_no_given_yes"] == 1.0
    assert values["yes_to_no_given_yes_difference"] == 0.5


def test_conditional_rates_are_nan_when_no_start_is_answered_yes():
    all_no = {pair_id: {variant: -1.0 for variant in margins} for pair_id, margins in MARGINS.items()}

    values = control_values(replacement_quads(_rows(margins=all_no))["cat"])

    assert values["n_start_yes"] == 0
    assert math.isnan(values["pp_yes_to_no_given_yes"])
    assert math.isnan(values["yes_to_no_given_yes_difference"])
    assert values["pn_Yes->No"] == 0.0  # 전체 출발 기준의 비율은 그대로 계산된다
    assert values["contrast_positive_rate"] == 0.5  # 모든 margin이 같으면 대비가 모두 0이다


def test_control_values_of_one_slot_use_only_that_quad():
    slot_3 = [quad for quad in replacement_quads(_rows())["cat"] if quad["slot"] == 3]

    values = control_values(slot_3)

    assert values["n_starts"] == 2
    assert values["contrast_positive_rate"] == 1.0  # 대비 2.0, 4.0


def test_leave_one_image_out_drops_whole_quads():
    # 이미지 1 ~ 4는 모든 묶음에 있어서 건너뛴다. 91을 빼면 자리 2, 1의 묶음이, 92를 빼면 자리 1의 묶음만,
    # 93을 빼면 자리 3의 묶음만 남는다
    ranges = leave_one_image_out(replacement_quads(_rows())["cat"], control_values)

    assert ranges["n_starts"] == (2, 4)
    # 자리 2와 1: 대비 0.0, 2.0, 2.5, 1.5 -> (3 + 0.5) / 4. 자리 1만, 자리 3만: 모두 양수
    assert ranges["contrast_positive_rate"] == (0.875, 1.0)


# ------------------------------------------------------------ dominance_verdict


def test_dominance_verdict_uses_the_fixed_band_around_one_half():
    assert dominance_verdict(0.56, 0.90) == "지지"  # 구간 하한이 0.55를 넘는다
    assert dominance_verdict(0.46, 0.54) == "반증"  # 구간 전체가 0.45 ~ 0.55 안: 두 교체가 구분되지 않는다
    assert dominance_verdict(0.50, 0.70) == "미결정"
    assert dominance_verdict(0.55, 0.90) == "미결정"  # 하한이 0.55와 같으면 넘은 것이 아니다
