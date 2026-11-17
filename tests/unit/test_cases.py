"""
src/mivlm/data/cases.py의 pilot_cases 단위 테스트
pytest tests/unit/test_cases.py -v
"""
from mivlm.data.cases import pilot_cases

VARIANTS = ["PPP", "PPN", "PNN", "NNN", "NPP", "PNP"]
CLASSES = ["person", "dining table", "chair", "couch", "cat", "train", "bus"]


def _make_cases():
    # 클래스 7개 x group 2개 x pair 2개 x 변형 6개. cases.jsonl과 같은 순서
    cases = []
    for name in CLASSES:
        for group in [1, 2]:
            group_id = f"group_{group:02d}_{name.replace(' ', '_')}"
            for pair_id in [1, 2]:
                for variant in VARIANTS:
                    cases.append({"case_id": f"{group_id}_{pair_id}_{variant}", "group_id": group_id,
                                  "pair_id": pair_id, "object": name, "variant": variant})
    return cases


def test_one_case_per_class_in_class_order():
    picked = pilot_cases(_make_cases())

    assert [case["object"] for case in picked] == CLASSES


def test_variants_rotate_with_class_order():
    picked = pilot_cases(_make_cases())

    assert [case["variant"] for case in picked] == ["PPP", "PPN", "PNN", "NNN", "NPP", "PNP", "PPP"]


def test_all_from_first_pair_of_first_group():
    for case in pilot_cases(_make_cases()):
        assert case["group_id"].startswith("group_01_")
        assert case["pair_id"] == 1
