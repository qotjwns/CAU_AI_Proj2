"""
src/mivlm/analysis/logit_bias.py 단위 테스트 (모델도, 저장된 결과도 쓰지 않는다)
pytest tests/unit/test_logit_bias.py -v
"""
import math

import pytest

from mivlm.analysis.logit_bias import (
    assign_biases, best_bias, bias_metrics, class_metrics, damage_limited_biases, largest_bias_within_damage,
    no_shift_candidates, selection_rules, sweep, tradeoff_curve, two_way_candidates,
)


def _rows(ppp: list[float], one_n: list[float], name: str = "cat", group: str = "g1") -> list[dict]:
    """PPP의 margin 목록과 N 1장 케이스의 margin 목록으로 행을 만든다. 답은 margin의 부호다 (0이면 No)"""
    margins = [("PPP", margin) for margin in ppp] + [("PPN", margin) for margin in one_n]
    return [{"object": name, "group_id": group, "variant": variant, "logit_margin": margin,
             "prediction": "Yes" if margin > 0 else "No"} for variant, margin in margins]


# PPP 4개와 N 1장 6개. 원래 답: PPP는 2개가 Yes, N 1장은 3개가 Yes(False-Yes)
MIXED = _rows(ppp=[2.0, 0.5, 0.0, -1.0], one_n=[1.0, 0.25, 0.0, -0.5, -2.0, 3.0])


# ---------------------------------------------------------------- 후보 b


def test_no_shift_candidates_are_0_and_every_positive_margin_in_increasing_order():
    # MIXED의 margin: PPP 2.0, 0.5, 0.0, -1.0 / N 1장 1.0, 0.25, 0.0, -0.5, -2.0, 3.0
    assert no_shift_candidates(MIXED) == [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]  # 0은 한 번만, 음수는 없다


def test_two_way_candidates_add_the_negative_margins():
    candidates = two_way_candidates(MIXED)

    assert candidates == [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
    assert [bias for bias in candidates if bias >= 0] == no_shift_candidates(MIXED)


def test_candidates_keep_float32_margins_without_rounding():
    margin = 0.3607063293457031  # float32 실행에 실제로 있는 모양의 값. 0.125 격자 위에 있지 않다
    rows = _rows(ppp=[margin], one_n=[-0.3414726257324219])

    assert no_shift_candidates(rows) == [0.0, margin]
    assert two_way_candidates(rows) == [-0.3414726257324219, 0.0, margin]


def test_candidates_between_grid_points_keep_the_damage_limit_where_a_0_125_grid_could_not():
    # 고친 문제의 재현: 원래 맞힌 PPP 20개 중 0.03과 0.11이 작다. False-Yes 둘은 0.07, 0.09이다.
    # 0.125 간격의 격자라면 0.125에서 이미 2/20 = 10%가 손상돼, 5% 한도에서 b = 0밖에 남지 않는다
    rows = _rows(ppp=[0.03, 0.11] + [2.0] * 18, one_n=[0.07, 0.09])

    bias = largest_bias_within_damage(rows, no_shift_candidates(rows), 0.05)

    assert bias == 0.09  # 0.11부터는 손상이 10%다
    metrics = bias_metrics(rows, [bias] * len(rows))
    assert (metrics["n_damaged"], metrics["ppp_damage_rate"]) == (1, 0.05)
    assert (metrics["n_corrected"], metrics["false_yes_correction_rate"]) == (2, 1.0)


# ---------------------------------------------------------------- bias_metrics


def test_a_bias_of_0_reproduces_the_original_answers():
    metrics = bias_metrics(MIXED, [0.0] * len(MIXED))

    original_yes = [row["prediction"] == "Yes" for row in MIXED]
    assert metrics["ppp_accuracy"] == sum(original_yes[:4]) / 4 == 0.5
    assert metrics["false_yes_rate"] == sum(original_yes[4:]) / 6 == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["ppp_accuracy_change"] == 0.0
    assert (metrics["n_ppp_yes"], metrics["n_damaged"], metrics["ppp_damage_rate"]) == (2, 0, 0.0)
    assert (metrics["n_false_yes"], metrics["n_corrected"], metrics["false_yes_correction_rate"]) == (3, 0, 0.0)


def test_raising_the_bias_turns_answers_to_no_and_counts_damage_and_correction_separately():
    metrics = bias_metrics(MIXED, [0.5] * len(MIXED))

    # PPP: 0.5보다 큰 것은 2.0뿐이다. 원래 Yes였던 0.5가 No가 됐다
    assert metrics["ppp_accuracy"] == 0.25
    assert metrics["ppp_accuracy_change"] == -0.25  # 전체 PPP 4개 기준의 변화
    assert (metrics["n_damaged"], metrics["ppp_damage_rate"]) == (1, 0.5)  # 원래 맞힌 2개 중 1개
    # N 1장: 0.5보다 큰 것은 1.0, 3.0이다. 원래 False-Yes였던 0.25가 No가 됐다
    assert metrics["false_yes_rate"] == pytest.approx(2 / 6)  # N 1장 전체 6개 중
    assert (metrics["n_corrected"], metrics["false_yes_correction_rate"]) == (1, pytest.approx(1 / 3))  # 원래 3개 중
    assert metrics["balanced_accuracy"] == pytest.approx((0.25 + 4 / 6) / 2)


def test_a_margin_equal_to_the_bias_is_answered_no():
    metrics = bias_metrics(_rows(ppp=[1.0], one_n=[1.0]), [1.0, 1.0])

    assert metrics["ppp_accuracy"] == 0.0
    assert metrics["false_yes_rate"] == 0.0


def test_each_row_can_have_its_own_bias():
    rows = _rows(ppp=[1.0, 1.0], one_n=[1.0, 1.0])

    metrics = bias_metrics(rows, [0.0, 2.0, 0.0, 2.0])  # 둘째 PPP와 둘째 N 1장에만 높은 기준

    assert metrics["ppp_accuracy"] == 0.5
    assert metrics["false_yes_rate"] == 0.5


def test_a_negative_bias_adds_yes_answers_and_damages_nothing():
    metrics = bias_metrics(MIXED, [-0.75] * len(MIXED))

    assert metrics["ppp_accuracy"] == 0.75  # 0.0도 Yes가 된다. -1.0은 그대로 No
    assert metrics["false_yes_rate"] == pytest.approx(5 / 6)  # -2.0만 No
    assert (metrics["n_damaged"], metrics["n_corrected"]) == (0, 0)


def test_rates_without_any_matching_case_are_nan():
    metrics = bias_metrics(_rows(ppp=[-1.0], one_n=[-1.0]), [0.5, 0.5])  # 원래 Yes인 케이스가 없다

    assert math.isnan(metrics["ppp_damage_rate"])
    assert math.isnan(metrics["false_yes_correction_rate"])
    assert metrics["false_yes_rate"] == 0.0


# ----------------------------------------------------------------------- sweep


def test_sweep_gives_the_same_values_as_bias_metrics_for_every_candidate():
    grid = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]

    swept = sweep(MIXED, grid)

    for position, bias in enumerate(grid):
        metrics = bias_metrics(MIXED, [bias] * len(MIXED))
        assert swept["balanced_accuracy"][position].item() == pytest.approx(metrics["balanced_accuracy"])
        assert swept["ppp_damage_rate"][position].item() == pytest.approx(metrics["ppp_damage_rate"])


# ------------------------------------------- best_bias, largest_bias_within_damage


def test_best_bias_is_the_candidate_closest_to_0_among_those_with_the_highest_balanced_accuracy():
    # b가 1.0 이상 2.0 미만이면 PPP 둘은 Yes, N 1장 넷은 No로 모두 맞는다. 후보(0, 0.5, 1.0, 2.0, 3.0) 중에는 1.0뿐이다
    rows = _rows(ppp=[3.0, 2.0], one_n=[1.0, 0.5, -1.0, -2.0])

    assert best_bias(rows, no_shift_candidates(rows)) == 1.0


def test_best_bias_stays_at_0_when_moving_does_not_help():
    rows = _rows(ppp=[1.0], one_n=[-1.0])  # b가 -1.0 이상 1.0 미만이면 모두 맞는다. 후보 -1.0과 0 중 0에 가까운 0

    assert best_bias(rows, no_shift_candidates(rows)) == 0.0
    assert best_bias(rows, two_way_candidates(rows)) == 0.0


def test_only_two_way_candidates_can_pick_a_negative_bias():
    # PPP가 모두 0 아래다. b가 -2.0 이상 -0.5 미만이면 PPP는 Yes, N 1장은 No로 모두 맞는다.
    # 후보는 margin 값이라 그 범위의 아래쪽 끝 -2.0을 고른다 (음수 쪽에서는 0에서 가장 먼 쪽)
    rows = _rows(ppp=[-0.5, -0.25], one_n=[-2.0, -3.0])

    assert best_bias(rows, two_way_candidates(rows)) == -2.0
    assert best_bias(rows, no_shift_candidates(rows)) == 0.0  # 0 이상에서는 어떤 기준도 균형 정확도 0.5를 바꾸지 못한다


def test_largest_bias_within_damage_is_the_highest_candidate_that_keeps_the_limit():
    # 원래 맞힌 PPP 5개: 기준 0.125에서 1개(20%), 0.25에서 2개(40%), 0.5에서 3개(60%)가 손상된다
    rows = _rows(ppp=[0.125, 0.25, 0.5, 1.0, 2.0], one_n=[0.3])
    candidates = no_shift_candidates(rows)  # 0, 0.125, 0.25, 0.3, 0.5, 1.0, 2.0

    assert largest_bias_within_damage(rows, candidates, 0.10) == 0.0  # 10% 안에서는 올릴 수 없다
    assert largest_bias_within_damage(rows, candidates, 0.20) == 0.125
    assert largest_bias_within_damage(rows, candidates, 0.50) == 0.3  # 0.25와 0.5 사이의 후보는 N 1장의 0.3뿐이다 (40%)


# --------------------------------------------------------------- assign_biases


def _mean_margin(rows: list[dict]) -> float:
    """기준을 고르는 함수 대신 쓰는 것: 어떤 행들로 골랐는지가 값에 드러난다"""
    return sum(row["logit_margin"] for row in rows) / len(rows)


def test_with_hold_out_a_unit_gets_the_bias_chosen_from_the_other_units():
    rows = _rows([1.0, 1.0], [], name="a") + _rows([5.0, 5.0, 5.0], [], name="b")

    biases = assign_biases(rows, lambda row: row["object"], lambda row: None, _mean_margin, hold_out=True)

    assert biases == [5.0, 5.0, 1.0, 1.0, 1.0]  # a는 b의 행으로, b는 a의 행으로 골랐다


def test_without_hold_out_a_unit_gets_the_bias_chosen_from_its_own_pool():
    rows = _rows([1.0, 1.0], [], name="a") + _rows([5.0, 5.0, 5.0], [], name="b")

    biases = assign_biases(rows, lambda row: row["object"], lambda row: row["object"], _mean_margin, hold_out=False)

    assert biases == [1.0, 1.0, 5.0, 5.0, 5.0]


def test_units_inside_a_pool_are_held_out_one_at_a_time_and_never_see_another_pool():
    rows = (_rows([1.0], [], "a", "g1") + _rows([3.0], [], "a", "g2") + _rows([5.0], [], "a", "g3")
            + _rows([100.0], [], "b", "g1") + _rows([200.0], [], "b", "g2"))

    biases = assign_biases(rows, lambda row: (row["object"], row["group_id"]), lambda row: row["object"], _mean_margin,
                           hold_out=True)

    assert biases == [4.0, 3.0, 2.0, 200.0, 100.0]  # a의 g1은 a의 g2, g3로 골랐다: (3 + 5) / 2


# ------------------------------------------- selection_rules, damage_limited_biases


def _two_classes() -> list[dict]:
    """클래스 a, b에 그룹 둘씩, 그룹마다 PPP 하나와 N 1장 하나"""
    return (_rows([3.0], [1.0], "a", "g1") + _rows([2.0], [0.5], "a", "g2")
            + _rows([0.5], [-1.0], "b", "g1") + _rows([0.25], [-2.0], "b", "g2"))


def test_the_four_rules_choose_the_bias_from_different_rows():
    rules = selection_rules(_two_classes(), {"a": "a", "b": "b"}, no_shift_candidates)

    assert rules["original"] == [0.0] * 8
    # global: a는 b의 행으로(0에서 이미 모두 맞음 -> 0.0), b는 a의 행으로(1.0 이상 2.0 미만에서 모두 맞음 -> 1.0).
    # b의 1.0은 a의 margin 값이다. b에는 1.0인 margin이 없다: 후보도 고르는 쪽의 행으로만 만든다
    assert rules["global"] == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    # class_cv: a의 g1은 a의 g2(PPP 2.0, N 0.5)로 -> 0.5, a의 g2는 a의 g1(PPP 3.0, N 1.0)로 -> 1.0. b는 어느 쪽이든 0.0
    assert rules["class_cv"] == [0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # class_oracle: a 전체(PPP 3.0, 2.0 / N 1.0, 0.5)로 -> 1.0, b 전체로 -> 0.0
    assert rules["class_oracle"] == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_global_holds_out_linked_classes_together():
    # c도 그룹이 둘이어야 한다. selection_rules는 class_cv도 같이 계산하고, 그것은 같은 클래스의 다른 그룹이 필요하다
    rows = _two_classes() + _rows([4.0], [2.5], "c", "g1") + _rows([4.0], [2.5], "c", "g2")

    rules = selection_rules(rows, {"a": 0, "b": 0, "c": 1}, no_shift_candidates)

    # a와 b는 c의 행(PPP 4.0, N 2.5)으로만 고른다 -> 2.5. c는 a와 b의 행으로 고른다
    assert rules["global"][:8] == [2.5] * 8
    assert rules["global"][8:] == [best_bias(_two_classes(), no_shift_candidates(_two_classes()))] * 4


def test_a_unit_with_no_other_rows_to_choose_from_is_an_error():
    rows = _two_classes() + _rows([4.0], [2.5], "c", "g1")  # c는 그룹이 하나뿐이라 class_cv를 정할 수 없다

    with pytest.raises(ValueError, match="고를 행이 없음"):
        selection_rules(rows, {"a": 0, "b": 0, "c": 1}, no_shift_candidates)


def test_the_damage_limit_is_chosen_outside_the_class_and_can_be_broken_inside_it():
    # a의 후보는 0, 1.5, 5.0이다. 1.5까지 올려도 a의 PPP는 손상되지 않고, 5.0에서는 전부 손상된다
    easy = _rows(ppp=[5.0], one_n=[1.5], name="a")
    fragile = _rows(ppp=[0.125, 0.25, 0.5, 1.0, 2.0], one_n=[0.3], name="b")  # b는 0.125에서 이미 20%가 손상된다
    rows = easy + fragile

    biases = damage_limited_biases(rows, {"a": "a", "b": "b"}, limit=0.20)

    # a에는 b에서 고른 기준, b에는 a에서 고른 기준. 1.5는 b에 없는 값이라, b의 margin이 후보에 쓰이지 않았음을 보여 준다
    assert biases == [0.125, 0.125] + [1.5] * 6
    by_class = class_metrics(rows, biases)
    assert by_class["a"]["ppp_damage_rate"] == 0.0
    assert by_class["b"]["ppp_damage_rate"] == 0.8  # 2.0만 Yes로 남는다. 평가한 쪽의 실제 손상률은 한도 20%를 넘는다


# ----------------------------------------------- class_metrics, tradeoff_curve


def test_class_metrics_are_computed_from_the_rows_of_each_class_only():
    rows = _rows([1.0, -1.0], [1.0], name="a") + _rows([2.0], [-1.0, -2.0], name="b")

    by_class = class_metrics(rows, [0.0] * len(rows))

    assert list(by_class) == ["a", "b"]
    assert (by_class["a"]["ppp_accuracy"], by_class["a"]["false_yes_rate"]) == (0.5, 1.0)
    assert (by_class["b"]["ppp_accuracy"], by_class["b"]["false_yes_rate"]) == (1.0, 0.0)


def test_the_tradeoff_curve_starts_at_the_original_answers_and_ends_with_everything_turned_to_no():
    curve = tradeoff_curve(MIXED)

    assert [point["bias"] for point in curve] == [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]  # 0과, 양수인 margin 값 전부
    assert (curve[0]["ppp_damage_rate"], curve[0]["false_yes_correction_rate"]) == (0.0, 0.0)
    assert (curve[-1]["ppp_damage_rate"], curve[-1]["false_yes_correction_rate"]) == (1.0, 1.0)
    damage = [point["ppp_damage_rate"] for point in curve]
    correction = [point["false_yes_correction_rate"] for point in curve]
    assert damage == sorted(damage) and correction == sorted(correction)  # 기준을 올리면 둘 다 줄지 않는다


def test_a_damage_rate_exactly_at_the_limit_counts_as_within_the_limit():
    # 원래 맞힌 PPP 20개 중 1개가 0.125이다. 후보 0.125와 0.3에서 손상률이 정확히 1/20 = 5%이고, 5.0에서는 100%다
    rows = _rows(ppp=[0.125] + [5.0] * 19, one_n=[0.3])

    assert largest_bias_within_damage(rows, no_shift_candidates(rows), 0.05) == 0.3
    assert sweep(rows, [0.125])["ppp_damage_rate"].item() == 0.05
