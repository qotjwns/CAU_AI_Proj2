"""
src/mivlm/analysis/position_weights.py 단위 테스트 (모델도, 저장된 결과도 쓰지 않는다)
pytest tests/unit/test_position_weights.py -v
"""
import math

import pytest
import torch

from mivlm.analysis.class_sensitivity import class_bootstrap_interval
from mivlm.analysis.position_weights import SUM_NAMES, slope_sums, slopes
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
