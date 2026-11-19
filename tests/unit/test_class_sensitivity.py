"""
src/mivlm/analysis/class_sensitivity.py 단위 테스트 (모델도, 저장된 결과도 쓰지 않는다)
pytest tests/unit/test_class_sensitivity.py -v
"""
import math

import pytest
import torch

from mivlm.analysis.class_sensitivity import (
    N_DRAWS, class_bootstrap_interval, class_pairs, class_values, difference_verdict, direction_verdict,
    explained_variance, leave_one_image_out, mean_interval, pair_images, rank_correlation, rank_correlation_interval,
    ranking_verdict, shared_images, without_images,
)


def _row(name: str, pair_id: int, variant: str, margin: float, image_ids: list[int]) -> dict:
    """A 실행의 행 하나. 답은 margin의 부호로 정한다 (양수면 Yes)"""
    return {"case_id": f"group_01_{name}_{pair_id}_{variant}", "group_id": f"group_01_{name}", "pair_id": pair_id,
            "object": name, "variant": variant, "image_ids": image_ids, "label": "Yes" if variant == "PPP" else "No",
            "prediction": "Yes" if margin > 0 else "No", "logit_margin": margin}


def _cat_rows() -> list[dict]:
    """cat 클래스의 pair 두 개. P 이미지는 1, 2, 3, 4이고 N 이미지는 pair 1이 9, pair 2가 8이다

    pair 1: PPP +2.0(Yes) / PPN -1.0(No) / PNP +0.5(Yes) / NPP +2.0(Yes)
    pair 2: PPP -0.5(No)  / PPN -1.5(No) / PNP +1.0(Yes) / NPP -0.5(No)
    """
    return [
        _row("cat", 1, "PPP", 2.0, [1, 2, 3]), _row("cat", 1, "PPN", -1.0, [1, 2, 9]),
        _row("cat", 1, "PNP", 0.5, [1, 9, 3]), _row("cat", 1, "NPP", 2.0, [9, 2, 3]),
        _row("cat", 2, "PPP", -0.5, [1, 2, 4]), _row("cat", 2, "PPN", -1.5, [1, 2, 8]),
        _row("cat", 2, "PNP", 1.0, [1, 8, 4]), _row("cat", 2, "NPP", -0.5, [8, 2, 4]),
    ]


def _dog_rows() -> list[dict]:
    """dog 클래스의 pair 하나. N 이미지 9를 cat과 같이 쓴다"""
    return [_row("dog", 1, "PPP", 1.0, [5, 6, 7]), _row("dog", 1, "PPN", 0.0, [5, 6, 9]),
            _row("dog", 1, "PNP", 0.0, [5, 9, 7]), _row("dog", 1, "NPP", 0.0, [9, 6, 7])]


# ---------------------------------------------------------------- class_pairs


def test_class_pairs_groups_the_four_variants_of_each_pair_by_class():
    extra = [_row("cat", 1, "PNN", -3.0, [1, 8, 9]), _row("cat", 1, "NNN", -5.0, [7, 8, 9])]  # 쓰지 않는 변형

    pairs_by_class = class_pairs(_cat_rows() + _dog_rows() + extra)

    assert list(pairs_by_class) == ["cat", "dog"]
    assert [pair["pair"] for pair in pairs_by_class["cat"]] == ["group_01_cat_1", "group_01_cat_2"]
    assert list(pairs_by_class["cat"][0]["cases"]) == ["PPP", "PPN", "PNP", "NPP"]  # PNN, NNN은 들어가지 않는다
    assert pairs_by_class["dog"][0]["cases"]["PPN"]["logit_margin"] == 0.0


# --------------------------------------------------------------- class_values


def test_class_values_of_ppp_and_of_one_variant_are_hand_computed():
    values = class_values(class_pairs(_cat_rows())["cat"])

    assert values["n_pairs"] == 2
    assert values["yes_rate_PPP"] == 0.5  # +2.0은 Yes, -0.5는 No
    assert values["mean_margin_PPP"] == 0.75
    # PPN: margin -1.0, -1.5. 둘 다 No
    assert values["yes_rate_PPN"] == 0.0
    assert values["mean_margin_PPN"] == -1.25
    assert values["yes_rate_drop_PPN"] == 0.5
    assert values["margin_drop_PPN"] == 2.0  # (2.0 - (-1.0)) = 3.0, (-0.5 - (-1.5)) = 1.0의 평균
    assert values["margin_drop_positive_rate_PPN"] == 1.0
    assert [values[f"{t}_PPN"] for t in ["Yes->No", "Yes->Yes", "No->No", "No->Yes"]] == [0.5, 0.0, 0.5, 0.0]
    assert values["auroc_PPN"] == 1.0  # PPP의 margin 2.0, -0.5가 PPN의 -1.0, -1.5보다 모두 높다


def test_margin_drop_can_be_zero_or_negative_and_only_strict_drops_count_as_positive():
    values = class_values(class_pairs(_cat_rows())["cat"])

    # PNP: margin +0.5, +1.0. 둘 다 Yes. 하락은 2.0 - 0.5 = 1.5, -0.5 - 1.0 = -1.5
    assert values["yes_rate_drop_PNP"] == -0.5  # Yes 비율이 0.5에서 1.0으로 올랐다
    assert values["margin_drop_PNP"] == 0.0
    assert values["margin_drop_positive_rate_PNP"] == 0.5
    assert [values[f"{t}_PNP"] for t in ["Yes->No", "Yes->Yes", "No->No", "No->Yes"]] == [0.0, 0.5, 0.0, 0.5]
    assert values["auroc_PNP"] == 0.5  # 2.0은 둘보다 높고, -0.5는 둘보다 낮다
    # NPP: margin +2.0, -0.5로 PPP와 같다. 하락은 둘 다 0이고, 0은 "하락"으로 세지 않는다
    assert values["margin_drop_NPP"] == 0.0
    assert values["margin_drop_positive_rate_NPP"] == 0.0
    assert [values[f"{t}_NPP"] for t in ["Yes->No", "Yes->Yes", "No->No", "No->Yes"]] == [0.0, 0.5, 0.5, 0.0]
    assert values["auroc_NPP"] == 0.5  # 같은 값 2쌍(반씩), 높은 것 1쌍, 낮은 것 1쌍: (0.5 + 0.5 + 1 + 0) / 4


def test_one_n_values_pool_the_three_variants():
    values = class_values(class_pairs(_cat_rows())["cat"])

    # 세 변형의 케이스 6개: margin -1.0, -1.5, +0.5, +1.0, +2.0, -0.5. Yes는 3개
    assert values["yes_rate_1N"] == 0.5
    assert values["mean_margin_1N"] == pytest.approx(0.5 / 6)
    assert values["yes_rate_drop_1N"] == 0.0
    assert values["margin_drop_1N"] == pytest.approx(4.0 / 6)  # 하락 3.0, 1.0, 1.5, -1.5, 0, 0의 평균
    assert values["margin_drop_positive_rate_1N"] == 0.5  # 3.0, 1.0, 1.5
    # pair 1(PPP는 Yes): PPN No, PNP Yes, NPP Yes / pair 2(PPP는 No): PPN No, PNP Yes, NPP No
    transitions = [values[f"{t}_1N"] for t in ["Yes->No", "Yes->Yes", "No->No", "No->Yes"]]
    assert transitions == pytest.approx([1 / 6, 2 / 6, 2 / 6, 1 / 6])
    assert values["yes_rate_drop_1N"] == pytest.approx(values["Yes->No_1N"] - values["No->Yes_1N"])
    # PPP 2개 대 6개, 12쌍: 2.0은 5개보다 높고 1개와 같다(5.5), -0.5는 2개보다 높고 1개와 같다(2.5)
    assert values["auroc_1N"] == pytest.approx(8 / 12)


# ------------------------------ pair_images, shared_images, without_images


def test_pair_images_are_all_images_of_the_four_cases():
    pairs = class_pairs(_cat_rows())["cat"]

    assert pair_images(pairs[0]) == {1, 2, 3, 9}
    assert pair_images(pairs[1]) == {1, 2, 4, 8}


def test_shared_images_are_those_used_by_two_classes():
    assert shared_images(class_pairs(_cat_rows() + _dog_rows())) == {9}
    assert shared_images(class_pairs(_cat_rows())) == set()  # 한 클래스 안에서 여러 번 쓰인 이미지는 해당하지 않는다


def test_without_images_drops_whole_pairs_so_that_ppp_and_its_variants_stay_together():
    pairs = class_pairs(_cat_rows())["cat"]

    kept = without_images(pairs, {9})  # 이미지 9는 pair 1의 PPN, PNP, NPP에만 있지만 pair 1 전체를 뺀다

    assert [pair["pair"] for pair in kept] == ["group_01_cat_2"]
    assert list(kept[0]["cases"]) == ["PPP", "PPN", "PNP", "NPP"]
    assert without_images(pairs, {1}) == []  # 이미지 1은 두 pair 모두에 있다
    assert without_images(pairs, {100}) == pairs


# ---------------------------------------------------------- leave_one_image_out


def test_leave_one_image_out_gives_the_range_over_the_images_that_leave_some_pair():
    # 이미지 1, 2를 빼면 pair가 남지 않아 건너뛴다. 3이나 9를 빼면 pair 2만, 4나 8을 빼면 pair 1만 남는다
    ranges = leave_one_image_out(class_pairs(_cat_rows())["cat"])

    assert ranges["n_pairs"] == (1, 1)
    assert ranges["margin_drop_PPN"] == (1.0, 3.0)  # pair 2만: 1.0, pair 1만: 3.0
    assert ranges["yes_rate_PPP"] == (0.0, 1.0)  # pair 2만: PPP가 No, pair 1만: PPP가 Yes


# ------------------------------------------- class_bootstrap_interval, mean_interval


def test_bootstrap_interval_of_identical_classes_is_that_value():
    assert mean_interval([0.3] * 20) == pytest.approx((0.3, 0.3, 0.3))


def test_bootstrap_draws_whole_classes_and_is_reproducible():
    values = torch.arange(20, dtype=torch.float64)
    seen = {}

    def mean_of_classes(drawn: torch.Tensor) -> torch.Tensor:
        seen["shape"] = drawn.shape
        return drawn.mean(dim=1)

    first = class_bootstrap_interval(values, mean_of_classes, 0.95)
    second = class_bootstrap_interval(values, mean_of_classes, 0.95)

    assert seen["shape"] == (N_DRAWS, 20)  # 뽑기마다 클래스 20개를 다시 뽑는다
    assert torch.equal(first, second)  # seed가 고정되어 있다
    assert first.shape == (2,)
    assert 0 <= first[0] < 9.5 < first[1] <= 19  # 0 ~ 19의 평균은 9.5


def test_a_higher_level_gives_a_wider_interval():
    per_class = [float(k) for k in range(20)]

    _, low_95, high_95 = mean_interval(per_class)
    _, low_983, high_983 = mean_interval(per_class, 1 - 0.05 / 3)

    assert low_983 < low_95 < high_95 < high_983


def test_mean_interval_returns_the_mean_of_the_classes():
    mean, low, high = mean_interval([1.0, 2.0, 6.0])

    assert mean == 3.0
    assert 1.0 <= low <= mean <= high <= 6.0


# ------------------------------------- rank_correlation, rank_correlation_interval


def test_rank_correlation_is_1_for_the_same_order_and_minus_1_for_the_reverse():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])

    assert rank_correlation(x, torch.tensor([10.0, 20.0, 30.0, 400.0])).item() == pytest.approx(1.0)
    assert rank_correlation(x, torch.tensor([4.0, 3.0, 2.0, 1.0])).item() == pytest.approx(-1.0)


def test_rank_correlation_gives_tied_values_their_mean_rank():
    # x의 순위 [1.5, 1.5, 3], y의 순위 [1, 2, 3]. 평균을 뺀 값은 [-0.5, -0.5, 1]과 [-1, 0, 1]
    # 곱의 합 1.5, 제곱합 1.5와 2 -> 1.5 / sqrt(3)
    x, y = torch.tensor([1.0, 1.0, 2.0]), torch.tensor([1.0, 2.0, 3.0])

    assert rank_correlation(x, y).item() == pytest.approx(1.5 / math.sqrt(3))


def test_rank_correlation_is_computed_for_each_leading_row_and_is_nan_without_any_difference():
    x = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    y = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])

    result = rank_correlation(x, y)

    assert result[:2].tolist() == pytest.approx([1.0, -1.0])
    assert math.isnan(result[2].item())  # x가 모두 같으면 순위에 차이가 없다


def test_rank_correlation_interval_returns_the_correlation_and_an_interval_around_it():
    x = [float(k) for k in range(20)]
    y = [2.0 * value for value in x]  # 순서가 완전히 같다

    assert rank_correlation_interval(x, y) == pytest.approx((1.0, 1.0, 1.0))


# --------------------------------------------------------- explained_variance


def test_explained_variance_is_the_share_of_margin_variance_between_groups():
    # 클래스 a의 margin 1, 3 (평균 2), 클래스 b의 margin 5, 7 (평균 6). 전체 평균 4
    # 전체 제곱합 9 + 1 + 1 + 9 = 20, 묶음 안의 제곱합 1 + 1 + 1 + 1 = 4 -> 1 - 4 / 20
    rows = [{"object": name, "variant": variant, "logit_margin": margin}
            for name, variant, margin in [("a", "PPP", 1.0), ("a", "PPN", 3.0), ("b", "PPP", 5.0), ("b", "PPN", 7.0)]]

    assert explained_variance(rows, ["object"]) == pytest.approx(0.8)
    assert explained_variance(rows, ["object", "variant"]) == pytest.approx(1.0)  # 묶음마다 행이 하나다


# ------------------------------------------------------------------- 판정


def test_direction_verdict_needs_the_whole_interval_on_one_side_of_zero():
    assert direction_verdict(0.1, 0.5) == "지지"
    assert direction_verdict(-0.5, -0.1) == "반증"
    assert direction_verdict(-0.1, 0.5) == "미결정"
    assert direction_verdict(0.0, 0.5) == "미결정"  # 0에 닿으면 지지가 아니다


def test_difference_verdict_supports_a_difference_in_either_direction_and_never_equality():
    assert difference_verdict(0.1, 0.5) == "지지"
    assert difference_verdict(-0.5, -0.1) == "지지"
    assert difference_verdict(-0.1, 0.1) == "미결정"


def test_ranking_verdict_uses_the_fixed_no_difference_band():
    assert ranking_verdict(0.46, 0.54) == "무차이 지지"  # 구간 전체가 0.45 ~ 0.55 안
    assert ranking_verdict(0.56, 0.90) == "무차이 기각"  # 하한이 0.55를 넘는다
    assert ranking_verdict(0.50, 0.60) == "미결정"  # 범위에 걸쳐 있다
    assert ranking_verdict(0.55, 0.90) == "미결정"  # 하한이 0.55와 같으면 넘은 것이 아니다
    assert ranking_verdict(0.30, 0.40) == "미결정"  # 범위 아래쪽은 기준에 없으므로 미결정이다
