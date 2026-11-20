"""
src/mivlm/analysis/position_weights.py 단위 테스트 (모델도, 저장된 결과도 쓰지 않는다)
pytest tests/unit/test_position_weights.py -v
"""
import math

import pytest
import torch

from mivlm.analysis.class_sensitivity import class_bootstrap_interval
from mivlm.analysis.position_weights import (
    SLOT_VARIANTS, SUM_NAMES, TERM_NAMES, shares, slope_sums, slopes, weight_statistics, weight_terms,
)
from mivlm.evaluation.metrics import ONE_N_VARIANTS


def _class(margins: list[tuple[float, float, float, float]]) -> list[dict]:
    """한 클래스의 pair 목록. margins의 항목 하나가 pair 하나의 (PPP, PPN, PNP, NPP) margin이다"""
    return [{"pair": f"pair_{index}", "cases": {variant: {"logit_margin": margin}
                                                 for variant, margin in zip(["PPP", *ONE_N_VARIANTS], values)}}
            for index, values in enumerate(margins)]


def _random_classes(sizes: list[int], seed: int = 0) -> dict[str, list[dict]]:
    """클래스마다 pair 수가 sizes인 무작위 margin. 클래스마다 평균이 다르게 offset을 더한다"""
    generator = torch.Generator().manual_seed(seed)
    classes = {}
    for index, size in enumerate(sizes):
        values = torch.randn(size, 4, generator=generator, dtype=torch.float64) + 2.0 * index
        classes[f"class_{index}"] = _class([tuple(row) for row in values.tolist()])
    return classes


def _xy(pairs_by_class: dict[str, list[dict]]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """직접 회귀에 쓸 값: x (pair 수,), y (pair 수, 변형 3개), 행마다의 클래스 번호"""
    x, y, labels = [], [], []
    for label, pairs in enumerate(pairs_by_class.values()):
        for pair in pairs:
            x.append(pair["cases"]["PPP"]["logit_margin"])
            y.append([pair["cases"][variant]["logit_margin"] for variant in ONE_N_VARIANTS])
            labels.append(label)
    return torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64), labels


def _least_squares_slope(columns: list[torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """columns(절편 또는 클래스 표시)와 x를 설명 변수로 둔 OLS에서 x의 계수: 변형마다, shape (3,)"""
    design = torch.stack([*columns, x], dim=1)
    return torch.linalg.lstsq(design, y).solution[-1]


# ---------------------------------------------------------------- slope_sums


def test_slope_sums_hold_one_row_per_class_with_the_variants_in_order():
    classes = {"a": _class([(1.0, 2.0, 3.0, 4.0), (3.0, 0.0, 1.0, 2.0)]), "b": _class([(5.0, 6.0, 7.0, 8.0)])}

    sums = slope_sums(classes, centered=False)

    assert sums.shape == (2, 3, 5) and sums.dtype == torch.float64
    assert SUM_NAMES == ["n", "x", "y", "xx", "xy"]
    # 클래스 a, 변형 PPN: x = (1, 3), y = (2, 0)
    assert sums[0, 0].tolist() == [2.0, 4.0, 2.0, 10.0, 2.0]
    # 클래스 a, 변형 NPP: x = (1, 3), y = (4, 2)
    assert sums[0, 2].tolist() == [2.0, 4.0, 6.0, 10.0, 10.0]
    assert sums[1, 1].tolist() == [1.0, 5.0, 7.0, 25.0, 35.0]


def test_centered_sums_subtract_each_class_mean():
    classes = {"a": _class([(1.0, 2.0, 3.0, 4.0), (3.0, 0.0, 1.0, 2.0)])}

    sums = slope_sums(classes, centered=True)

    # PPN: x - 평균 = (-1, 1), y - 평균 = (1, -1)
    assert sums[0, 0].tolist() == [2.0, 0.0, 0.0, 2.0, -2.0]


# ---------------------------------------------------------------- slopes


def test_the_pooled_slope_matches_least_squares_with_an_intercept():
    classes = _random_classes([6, 4, 5])
    x, y, _ = _xy(classes)

    expected = _least_squares_slope([torch.ones_like(x)], x, y)

    assert torch.allclose(slopes(slope_sums(classes, centered=False)), expected)


def test_the_within_class_slope_matches_least_squares_with_one_indicator_per_class():
    classes = _random_classes([6, 4, 5])
    x, y, labels = _xy(classes)
    indicators = [torch.tensor([float(label == number) for label in labels], dtype=torch.float64)
                  for number in range(len(classes))]

    expected = _least_squares_slope(indicators, x, y)

    assert torch.allclose(slopes(slope_sums(classes, centered=True)), expected)


def test_class_offsets_change_the_pooled_slope_but_not_the_within_class_slope():
    # 클래스 안에서는 y = 0.5 x + 클래스마다의 offset. x가 큰 클래스일수록 offset이 커서, 전체 회귀는 0.5보다 가파르다
    def make(xs, offset):
        return _class([(x, 0.5 * x + offset, 0.5 * x + offset, 0.5 * x + offset) for x in xs])

    classes = {"low": make([0.0, 1.0, 2.0], 0.0), "high": make([4.0, 5.0, 6.0], 3.0)}

    within = slopes(slope_sums(classes, centered=True))
    pooled = slopes(slope_sums(classes, centered=False))

    assert torch.allclose(within, torch.full((3,), 0.5, dtype=torch.float64))
    assert (pooled > 0.5).all()


def test_a_class_with_one_pair_adds_nothing_to_the_within_class_slope():
    classes = _random_classes([6, 4])
    with_single = {**classes, "single": _class([(9.0, -3.0, 5.0, 1.0)])}

    within_before = slopes(slope_sums(classes, centered=True))
    within_after = slopes(slope_sums(with_single, centered=True))

    assert torch.allclose(within_before, within_after)  # 클래스 평균을 빼면 그 pair는 0이 된다
    assert not torch.allclose(slopes(slope_sums(classes, centered=False)),
                              slopes(slope_sums(with_single, centered=False)))  # 전체 회귀에는 들어간다


def test_no_spread_in_the_ppp_margin_gives_nan():
    classes = {"a": _class([(1.0, 2.0, 3.0, 4.0), (1.0, 0.0, 1.0, 2.0)]), "b": _class([(1.0, 5.0, 5.0, 5.0)])}

    assert all(math.isnan(value) for value in slopes(slope_sums(classes, centered=False)).tolist())
    assert all(math.isnan(value) for value in slopes(slope_sums(classes, centered=True)).tolist())


# ------------------------------------------------- 클래스를 다시 뽑았을 때 (bootstrap)


def test_resampled_classes_give_the_slope_of_the_duplicated_data():
    # 클래스 0을 두 번, 클래스 1을 한 번 뽑은 경우. 충분통계를 뽑은 대로 더한 값이 데이터를 그대로 복제해 푼 값과 같아야 한다
    classes = _random_classes([6, 4, 5])
    names = list(classes)
    duplicated = {"copy_1": classes[names[0]], "copy_2": classes[names[0]], "other": classes[names[1]]}
    picked = torch.tensor([0, 0, 1])

    for centered in [False, True]:
        from_sums = slopes(slope_sums(classes, centered)[picked])
        direct = slopes(slope_sums(duplicated, centered))
        assert torch.allclose(from_sums, direct)

    x, y, labels = _xy(duplicated)
    indicators = [torch.tensor([float(label == number) for label in labels], dtype=torch.float64)
                  for number in range(3)]
    assert torch.allclose(slopes(slope_sums(classes, centered=True)[picked]), _least_squares_slope(indicators, x, y))


def test_slopes_accept_a_batch_of_draws():
    sums = slope_sums(_random_classes([6, 4, 5]), centered=True)
    draws = sums[torch.tensor([[0, 1, 2], [2, 2, 0]])]  # 뽑기 2번, 클래스 3개씩

    batched = slopes(draws)

    assert batched.shape == (2, 3)
    assert torch.allclose(batched[1], slopes(sums[torch.tensor([2, 2, 0])]))


def test_the_class_bootstrap_interval_of_the_slopes_is_reproducible():
    sums = slope_sums(_random_classes([6, 4, 5, 7]), centered=True)

    first = class_bootstrap_interval(sums, slopes, 0.95)
    second = class_bootstrap_interval(sums, slopes, 0.95)

    assert first.shape == (2, 3)
    assert torch.equal(first, second)  # seed가 고정되어 같은 입력이면 같은 구간
    assert (first[0] <= first[1]).all()


@pytest.mark.parametrize("centered", [False, True])
def test_variant_order_follows_one_n_variants(centered):
    # PPN만 x를 그대로 따르고(기울기 1), 나머지는 상수(기울기 0)인 데이터
    classes = {"a": _class([(float(x), float(x), 1.0, 1.0) for x in range(5)])}

    assert slopes(slope_sums(classes, centered)).tolist() == pytest.approx([1.0, 0.0, 0.0])


# ============================================================ 자리 가중치와 하락 몫 (7-A-2)


def _additive_classes(weights: tuple[float, float, float], n_shift: tuple[float, float, float],
                      sizes: tuple[int, ...] = (6, 5, 7), seed: int = 0) -> dict[str, list[dict]]:
    """완전한 가산 모형으로 만든 클래스들: 공동 margin = 클래스 offset + Σ_j w_j s_j (s_j는 자리 j의 단일 margin)

    N이 자리 k에 들어가면 s_k만 N의 단일 margin으로 바뀌고, 공동 margin도 같은 식으로 계산한다.
    P의 단일 margin은 평균 2 근처, 자리 k에 들어가는 N의 단일 margin은 평균 -n_shift[k] 근처다.
    그래서 자리마다 단일 margin의 변화량 Δ_k가 약 2 + n_shift[k]로 다르다.
    """
    generator = torch.Generator().manual_seed(seed)
    w = torch.tensor(weights, dtype=torch.float64)
    shift = torch.tensor(n_shift, dtype=torch.float64)
    classes = {}
    for number, size in enumerate(sizes):
        p_single = torch.randn(size, 3, generator=generator, dtype=torch.float64) + 2.0
        n_single = torch.randn(size, 3, generator=generator, dtype=torch.float64) - shift
        offset = float(number)
        pairs = []
        for row in range(size):
            cases = {"PPP": {"logit_margin": offset + (w @ p_single[row]).item(),
                             "single_margins": p_single[row].tolist()}}
            for slot, variant in enumerate(SLOT_VARIANTS):
                single = p_single[row].clone()
                single[slot] = n_single[row, slot]
                cases[variant] = {"logit_margin": offset + (w @ single).item(), "single_margins": single.tolist()}
            pairs.append({"pair": f"pair_{row}", "cases": cases})
        classes[f"class_{number}"] = pairs
    return classes


WEIGHTS = (0.2, 0.3, 0.5)
N_SHIFT = (1.0, 2.0, 6.0)  # 자리 3의 N이 단일 margin을 훨씬 크게 바꾼다


def test_slot_variants_are_in_slot_order():
    assert SLOT_VARIANTS == ["NPP", "PNP", "PPN"]  # 자리 1, 2, 3에 N이 들어간 변형


# ---------------------------------------------------------------- weight_terms


def test_weight_terms_hold_one_row_per_class_and_slot():
    classes = _additive_classes(WEIGHTS, N_SHIFT)

    terms = weight_terms(classes)

    assert terms.shape == (3, 3, len(TERM_NAMES)) and terms.dtype == torch.float64
    yy, y2 = TERM_NAMES.index("yy"), TERM_NAMES.index("y2")
    assert torch.equal(terms[:, 0, yy], terms[:, 2, yy]) and torch.equal(terms[:, 0, y2], terms[:, 1, y2])
    # 클래스 0, 자리 3(PPN)의 하락 항은 그 클래스의 m(PPP) - m(PPN)의 평균이다
    drops = [pair["cases"]["PPP"]["logit_margin"] - pair["cases"]["PPN"]["logit_margin"] for pair in classes["class_0"]]
    assert terms[0, 2, TERM_NAMES.index("drop")].item() == pytest.approx(sum(drops) / len(drops))


# ---------------------------------------------------------------- shares


def test_shares_divide_by_the_sum():
    assert shares(torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)).tolist() == [0.25, 0.25, 0.5]
    assert shares(torch.tensor([2.0, -1.0, 1.0], dtype=torch.float64)).tolist() == [1.0, -0.5, 0.5]  # 합이 양수면 정한다


@pytest.mark.parametrize("parts", [[1.0, -1.0, 0.0], [-1.0, -1.0, -2.0], [0.0, 0.0, 0.0], [1.0, -1.0 + 1e-14, 0.0]])
def test_shares_are_nan_unless_the_sum_is_clearly_positive(parts):
    assert all(math.isnan(value) for value in shares(torch.tensor(parts, dtype=torch.float64)).tolist())


# ---------------------------------------------------------------- weight_statistics


def test_weights_are_recovered_from_exact_additive_data():
    statistics = weight_statistics(weight_terms(_additive_classes(WEIGHTS, N_SHIFT)))

    assert torch.allclose(statistics["weights"], torch.tensor(WEIGHTS, dtype=torch.float64))


def test_exact_additive_data_fit_the_prediction_even_when_the_changes_differ_by_slot():
    statistics = weight_statistics(weight_terms(_additive_classes(WEIGHTS, N_SHIFT)))
    zero = torch.zeros(3, dtype=torch.float64)

    assert torch.allclose(statistics["share_gap"], zero, atol=1e-10)  # r = δ - δ̂ = 0
    assert torch.allclose(statistics["magnitude_ratio"], torch.ones(3, dtype=torch.float64))
    assert torch.allclose(statistics["slope_gap"], zero, atol=1e-10)  # 관측 기울기 = 가산 모형의 기울기
    # Δ가 자리마다 달라서 δ - ω는 0이 아니다. 그래서 δ - ω를 주 지표로 쓰지 않는다
    assert statistics["share_vs_weight"].abs().max() > 0.05
    # N의 단일 margin이 PPP margin과 공분산을 가지면, 남은 P만으로 낸 기울기는 관측 기울기와 다르다
    assert not torch.allclose(statistics["kept_slope"], statistics["observed_slope"])


def test_the_observed_slope_is_the_within_class_slope_of_7a1_in_slot_order():
    classes = _additive_classes(WEIGHTS, N_SHIFT)
    in_variant_order = slopes(slope_sums(classes, centered=True))  # PPN, PNP, NPP

    observed = weight_statistics(weight_terms(classes))["observed_slope"]  # 자리 1, 2, 3 = NPP, PNP, PPN

    assert torch.allclose(observed, in_variant_order[[ONE_N_VARIANTS.index(variant) for variant in SLOT_VARIANTS]])


def test_the_magnitude_ratio_accepts_a_negative_denominator():
    # 자리 1의 가중치가 음수라 w_1Δ_1 < 0이다. 그래도 Σw > 0이므로 몫은 정해지고, ρ도 정해진다
    statistics = weight_statistics(weight_terms(_additive_classes((-0.2, 0.3, 0.5), N_SHIFT)))

    assert (statistics["weights"][0] * statistics["change"][0]).item() < 0
    assert torch.allclose(statistics["magnitude_ratio"], torch.ones(3, dtype=torch.float64))
    assert statistics["weight_share"][0].item() == pytest.approx(-0.2 / 0.6)


def test_a_singular_weight_regression_gives_nan_instead_of_an_error():
    classes = _additive_classes(WEIGHTS, N_SHIFT)
    for pairs in classes.values():  # 자리 1과 2의 단일 margin이 항상 같으면 XᵀX가 특이하다
        for pair in pairs:
            single = pair["cases"]["PPP"]["single_margins"]
            single[1] = single[0]

    statistics = weight_statistics(weight_terms(classes))

    for name in ["weights", "weight_share", "predicted_share", "share_gap", "magnitude_ratio", "additive_slope"]:
        assert all(math.isnan(value) for value in statistics[name].tolist()), name


def test_no_variation_in_the_ppp_margin_gives_nan_slopes():
    statistics = weight_statistics(weight_terms(_additive_classes((0.0, 0.0, 0.0), N_SHIFT)))  # m = 클래스 offset

    for name in ["kept_slope", "additive_slope", "observed_slope", "weight_share", "predicted_share"]:
        assert all(math.isnan(value) for value in statistics[name].tolist()), name


def test_a_class_with_one_pair_does_not_change_the_weights():
    classes = _additive_classes(WEIGHTS, N_SHIFT)
    with_single = {**classes, "single": _additive_classes(WEIGHTS, N_SHIFT, sizes=(1,), seed=1)["class_0"]}

    before, after = weight_statistics(weight_terms(classes)), weight_statistics(weight_terms(with_single))

    assert torch.allclose(before["weights"], after["weights"])
    assert not torch.allclose(before["drop"], after["drop"])  # D와 Δ는 클래스 평균이라 그 클래스도 한 표를 갖는다


def test_resampled_classes_give_the_statistics_of_the_duplicated_data():
    classes = _additive_classes((0.4, 0.1, 0.7), N_SHIFT)
    duplicated = {"copy_1": classes["class_0"], "copy_2": classes["class_0"], "other": classes["class_2"]}

    from_terms = weight_statistics(weight_terms(classes)[torch.tensor([0, 0, 2])])
    direct = weight_statistics(weight_terms(duplicated))

    for name in from_terms:
        assert torch.allclose(from_terms[name], direct[name], equal_nan=True), name


def test_weight_statistics_accept_a_batch_of_draws():
    terms = weight_terms(_additive_classes(WEIGHTS, N_SHIFT))
    draws = terms[torch.tensor([[0, 1, 2], [2, 2, 1]])]  # 뽑기 2번

    batched = weight_statistics(draws)

    assert all(values.shape == (2, 3) for values in batched.values())
    assert torch.allclose(batched["share_gap"][1], weight_statistics(terms[torch.tensor([2, 2, 1])])["share_gap"])
