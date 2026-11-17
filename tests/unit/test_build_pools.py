"""
scripts/build_pools.py 단위 테스트
pytest tests/unit/test_build_pools.py -v
"""
from build_pools import find_candidates, select_classes

DOG, CAT = 1, 2


def _make_coco():
    # 모든 이미지는 100x100 (면적 10,000) → 5% 기준은 bbox 면적 500
    def ann(image_id, category_id, w, h, iscrowd=0):
        return {"image_id": image_id, "category_id": category_id, "bbox": [0, 0, w, h], "iscrowd": iscrowd}

    return {
        "images": [{"id": i, "width": 100, "height": 100} for i in range(1, 8)],
        "categories": [{"id": DOG, "name": "dog"}, {"id": CAT, "name": "cat"}],
        "annotations": [
            ann(1, DOG, 50, 50),               # 1: 큰 dog → P
            ann(2, DOG, 10, 10),               # 2: 작은 dog만 → 제외
            ann(3, CAT, 50, 50),               # 3: dog 없음 → dog의 N
            ann(4, DOG, 60, 60, iscrowd=1),    # 4: crowd dog만 → 제외
            ann(5, DOG, 10, 10),               # 5: 작은 dog + 큰 dog → P
            ann(5, DOG, 50, 50),
            ann(6, DOG, 25, 20),               # 6: 정확히 500 (5%) → P
                                               # 7: 주석 없음 → 모든 클래스의 N
        ],
    }


# ------------------------------------------------------------- find_candidates

def test_large_object_is_positive():
    positives, negatives = find_candidates(_make_coco())
    assert 1 in positives[DOG]
    assert 1 not in negatives[DOG]


def test_small_only_is_excluded():
    positives, negatives = find_candidates(_make_coco())
    assert 2 not in positives[DOG]
    assert 2 not in negatives[DOG]


def test_absent_class_is_negative():
    positives, negatives = find_candidates(_make_coco())
    assert 3 in negatives[DOG]
    assert 3 not in positives[DOG]


def test_crowd_only_is_excluded():
    positives, negatives = find_candidates(_make_coco())
    assert 4 not in positives[DOG]
    assert 4 not in negatives[DOG]


def test_one_large_among_small_is_positive():
    positives, _ = find_candidates(_make_coco())
    assert 5 in positives[DOG]


def test_exactly_five_percent_is_positive():
    positives, _ = find_candidates(_make_coco())
    assert 6 in positives[DOG]


def test_unannotated_image_is_negative_for_every_class():
    _, negatives = find_candidates(_make_coco())
    assert 7 in negatives[DOG]
    assert 7 in negatives[CAT]


# -------------------------------------------------------------- select_classes

def test_select_drops_classes_with_too_few_candidates():
    positives = {1: list(range(10)), 2: list(range(7)), 3: list(range(10))}
    negatives = {1: list(range(10)), 2: list(range(10)), 3: list(range(7))}

    assert select_classes(positives, negatives) == [1]


def test_select_orders_by_positive_count():
    positives = {1: list(range(8)), 2: list(range(12)), 3: list(range(10))}
    negatives = {cid: list(range(8)) for cid in positives}

    assert select_classes(positives, negatives) == [2, 3, 1]


def test_select_breaks_ties_by_category_id():
    positives = {5: list(range(10)), 3: list(range(10))}
    negatives = {cid: list(range(8)) for cid in positives}

    assert select_classes(positives, negatives) == [3, 5]


def test_select_keeps_top_20():
    positives = {cid: list(range(8 + cid)) for cid in range(1, 22)}
    negatives = {cid: list(range(8)) for cid in positives}

    selected = select_classes(positives, negatives)

    assert len(selected) == 20
    assert 1 not in selected
