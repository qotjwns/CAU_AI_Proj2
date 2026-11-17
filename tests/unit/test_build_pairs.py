"""
scripts/build_pairs.py 단위 테스트
pytest tests/unit/test_build_pairs.py -v
"""
from collections import Counter

from build_pairs import make_pairs

POSITIVES = list(range(10, 18))  # P0~P7 → 10~17
NEGATIVES = list(range(20, 28))  # N0~N7 → 20~27


def _make_pool(name="dog"):
    return {"category_id": 18, "name": name, "positives": POSITIVES, "negatives": NEGATIVES}


def _pairs_of_group(group_id):
    return [pair for pair in make_pairs(_make_pool()) if pair["group_id"] == group_id]


def test_twenty_pairs_per_pool():
    assert len(make_pairs(_make_pool())) == 20


def test_first_group_pairs_are_ordered_combinations():
    pairs = _pairs_of_group("group_01_dog")

    assert [pair["p"] for pair in pairs] == [[10, 11, 12], [10, 11, 13], [10, 12, 13], [11, 12, 13]]
    assert [pair["pair_id"] for pair in pairs] == [1, 2, 3, 4]


def test_group_uses_its_position_pattern():
    used = {image_id for pair in _pairs_of_group("group_03_dog") for image_id in pair["p"]}

    assert used == {10, 12, 14, 16}  # 패턴 (0, 2, 4, 6)


def test_each_positive_appears_in_two_or_three_groups():
    groups = {}
    for pair in make_pairs(_make_pool()):
        groups.setdefault(pair["group_id"], set()).update(pair["p"])
    counts = Counter(image_id for images in groups.values() for image_id in images)

    assert [counts[p] for p in POSITIVES] == [3, 2, 2, 3, 3, 2, 2, 3]


def test_negatives_are_three_distinct_from_pool():
    for pair in make_pairs(_make_pool()):
        assert len(set(pair["n"])) == 3
        assert set(pair["n"]) <= set(NEGATIVES)


def test_same_pool_gives_same_pairs():
    assert make_pairs(_make_pool()) == make_pairs(_make_pool())


def test_name_with_space():
    pairs = make_pairs(_make_pool(name="dining table"))

    assert pairs[0]["group_id"] == "group_01_dining_table"
    assert pairs[0]["object"] == "dining table"
