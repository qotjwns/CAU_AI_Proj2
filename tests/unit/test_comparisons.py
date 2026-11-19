"""
src/mivlm/analysis/comparisons.py 단위 테스트 (모델은 로딩하지 않는다)
pytest tests/unit/test_comparisons.py -v
"""
import json

import pytest
import torch

from mivlm.analysis.comparisons import (
    ONE_IMAGE_CHANGED, is_position_controlled, load_comparison_rows, median_by_layer, same_prefix_baseline,
)
from mivlm.analysis.pair_record import COMPARISONS
from mivlm.analysis.pair_store import save_pair_record


def _comparison(family: str, image: str, token_cosine: list[list[float]], same_token_index: bool,
                same_position_ids: bool) -> dict:
    """층 2개짜리 비교 하나. token_cosine: 층마다 토큰별 값"""
    return {
        "family": family, "image": image, "image_id": 21,
        "a": {"variant": "PPN", "slot": 3, "token_span": [9, 12], "position_start": 9},
        "b": {"variant": "PNN", "slot": 3, "token_span": [9, 12], "position_start": 8},
        "same_token_index": same_token_index, "same_position_ids": same_position_ids,
        "token_cosine": torch.tensor(token_cosine),
        "cosine_of_means": torch.tensor([1.0, 0.75]),
        "relative_l2": torch.tensor([0.0, 0.5]),
    }


@pytest.fixture
def run_dir(tmp_path):
    """pair 2개, pair마다 비교 2개가 저장된 실행 폴더"""
    records = [
        {"pair": "pair_a", "cases": {}, "comparisons": [
            _comparison("slot_change", "N1", [[1.0, 1.0, 1.0], [0.5, 0.75, 1.0]], False, False),
            _comparison("same_prefix", "P2", [[1.0, 1.0], [1.0, 1.0]], True, True),
        ]},
        {"pair": "pair_b", "cases": {}, "comparisons": [
            _comparison("slot_change", "N1", [[1.0, 1.0], [0.25, 0.75]], False, True),
            _comparison("same_prefix", "P2", [[1.0], [1.0]], True, True),
        ]},
    ]
    with open(tmp_path / "results.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps({"pair": record["pair"], **save_pair_record(tmp_path, record)}) + "\n")
    return tmp_path


# ------------------------------------------------------- load_comparison_rows


def test_rows_are_one_per_comparison_in_saved_order(run_dir):
    rows = load_comparison_rows(run_dir)

    assert [(row["pair"], row["index"]) for row in rows] == [("pair_a", 0), ("pair_a", 1), ("pair_b", 0), ("pair_b", 1)]
    assert [row["family"] for row in rows] == ["slot_change", "same_prefix", "slot_change", "same_prefix"]
    assert [row["image"] for row in rows] == ["N1", "P2", "N1", "P2"]


def test_rows_average_token_cosine_over_tokens_per_layer(run_dir):
    # pair마다 토큰 수가 달라도, 행에서는 층마다 값 하나가 된다
    rows = load_comparison_rows(run_dir)

    assert rows[0]["token_cosine"].tolist() == [1.0, 0.75]  # 층 1: (0.5 + 0.75 + 1.0) / 3
    assert rows[2]["token_cosine"].tolist() == [1.0, 0.5]   # 층 1: (0.25 + 0.75) / 2
    for row in rows:
        assert row["token_cosine"].shape == (2,)


def test_rows_keep_the_per_layer_metrics_and_position_conditions(run_dir):
    rows = load_comparison_rows(run_dir)

    assert rows[0]["cosine_of_means"].tolist() == [1.0, 0.75]
    assert rows[0]["relative_l2"].tolist() == [0.0, 0.5]
    assert [(row["same_token_index"], row["same_position_ids"]) for row in rows] == [
        (False, False), (True, True), (False, True), (True, True),
    ]


# ------------------------------------------------------------ median_by_layer


def test_median_is_taken_per_layer_across_rows():
    rows = [{"relative_l2": torch.tensor(values)} for values in [[0.0, 0.25], [0.0, 0.5], [0.0, 4.0]]]

    # 층 1에서 유난히 큰 값(4.0)에 끌려가지 않는다. 평균이었다면 약 1.58이다
    assert median_by_layer(rows, "relative_l2").tolist() == [0.0, 0.5]


def test_median_of_an_even_number_of_rows_is_the_middle_of_the_two():
    rows = [{"relative_l2": torch.tensor([value])} for value in [0.25, 0.5, 1.0, 4.0]]

    assert median_by_layer(rows, "relative_l2").tolist() == [0.75]


# ---------------------------------------------------- is_position_controlled


def test_controlled_only_when_token_index_and_position_ids_are_both_the_same():
    def row(same_token_index, same_position_ids):
        return {"same_token_index": same_token_index, "same_position_ids": same_position_ids}

    assert is_position_controlled(row(True, True))
    assert not is_position_controlled(row(True, False))
    assert not is_position_controlled(row(False, True))
    assert not is_position_controlled(row(False, False))


# ------------------------------------------------------- same_prefix_baseline


def test_baseline_is_the_largest_layer_median_of_the_same_prefix_comparison():
    # same_prefix는 COMPARISONS의 마지막(순번 11)이다. 다른 비교의 값은 기준선에 들어가지 않는다
    rows = [
        {"index": 11, "relative_l2": torch.tensor([0.0, 0.25])},
        {"index": 11, "relative_l2": torch.tensor([0.0, 0.5])},
        {"index": 11, "relative_l2": torch.tensor([0.0, 0.75])},
        {"index": 0, "relative_l2": torch.tensor([0.0, 9.0])},
    ]

    assert same_prefix_baseline(rows) == 0.5


# ---------------------------------------------------------- ONE_IMAGE_CHANGED


def test_one_image_changed_lists_exactly_the_comparisons_where_one_earlier_image_differs():
    # ONE_IMAGE_CHANGED는 순번으로 적혀 있어서, 뜻과 어긋나도 코드만 봐서는 티가 나지 않는다.
    # 그래서 COMPARISONS에서 규칙으로 다시 찾아 같은지 본다: 두 입력에서 그 이미지의 자리가 같고,
    # 그 앞자리 가운데 정확히 한 자리의 이미지가 다르다
    images = {  # 변형마다 자리 1, 2, 3의 이미지 (mivlm.analysis.pair_record의 설명과 같다)
        "PPP": ["P1", "P2", "P3"], "PPN": ["P1", "P2", "N1"], "PNP": ["P1", "N1", "P3"],
        "NPP": ["N1", "P2", "P3"], "PNN": ["P1", "N2", "N1"], "NNN": ["N3", "N2", "N1"],
    }
    expected = []
    for index, (_, _, (a_variant, a_slot), (b_variant, b_slot)) in enumerate(COMPARISONS):
        earlier = zip(images[a_variant][:a_slot - 1], images[b_variant][:b_slot - 1])
        if a_slot == b_slot and sum(a != b for a, b in earlier) == 1:
            expected.append(index)

    assert ONE_IMAGE_CHANGED == expected
