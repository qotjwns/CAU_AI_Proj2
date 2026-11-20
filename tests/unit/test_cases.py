"""
src/mivlm/data/cases.py 단위 테스트
pytest tests/unit/test_cases.py -v
"""
from mivlm.data.cases import (
    POOL_PILOT_VARIANTS, case_pairs, image_linked_classes, items_for_cases, per_image_queries, pilot_cases,
    single_image_items,
)

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


def test_new_pool_pilots_rotate_only_over_the_four_variants():
    picked = pilot_cases(_make_cases(), pool="dev")

    assert POOL_PILOT_VARIANTS == ["PPP", "PPN", "PNP", "NPP"]
    assert [case["object"] for case in picked] == CLASSES
    # 첫 pair에서 PNN, NNN을 빼고 파일 순서(PPP, PPN, NPP, PNP)대로 돌아간다
    assert [case["variant"] for case in picked] == ["PPP", "PPN", "NPP", "PNP", "PPP", "PPN", "NPP"]
    assert pilot_cases(_make_cases(), pool="dev") == picked  # 무작위가 없다


def test_the_exploration_data_pilot_is_unchanged_by_the_pool_argument():
    assert pilot_cases(_make_cases(), pool=None) == pilot_cases(_make_cases())


def test_items_for_cases_keeps_the_items_of_the_case_images_and_classes():
    items = [{"object": "cat", "image_id": 1}, {"object": "cat", "image_id": 2}, {"object": "dog", "image_id": 1},
             {"object": "cat", "image_id": 9}]
    cases = [{"object": "cat", "image_ids": [9, 1, 5]}]

    assert items_for_cases(items, cases) == [items[0], items[3]]  # dog의 이미지 1은 클래스가 달라 빠진다


# ---------------------------------------------------------- single_image_items


def test_single_image_items_positives_then_negatives_per_class():
    # 이미지 1은 dining table의 P이면서 dog의 N이다. 질문이 다르므로 서로 다른 항목이어야 한다
    pools = [
        {"category_id": 67, "name": "dining table", "positives": [1, 2], "negatives": [3]},
        {"category_id": 18, "name": "dog", "positives": [4], "negatives": [5, 1]},
    ]

    items = single_image_items(pools)

    assert [item["item_id"] for item in items] == ["dining_table_1", "dining_table_2", "dining_table_3", "dog_4", "dog_5", "dog_1"]
    assert [item["label"] for item in items] == ["Yes", "Yes", "No", "Yes", "No", "No"]
    assert items[0] == {"item_id": "dining_table_1", "category_id": 67, "object": "dining table", "image_id": 1, "label": "Yes"}


# ----------------------------------------------------------- per_image_queries


def test_per_image_queries_one_per_slot_with_slot_label():
    case = {"case_id": "group_01_dog_1_PNP", "object": "dog", "variant": "PNP", "image_ids": [11, 21, 13]}

    queries = per_image_queries([case])

    assert [query["query_id"] for query in queries] == ["group_01_dog_1_PNP_1", "group_01_dog_1_PNP_2", "group_01_dog_1_PNP_3"]
    assert [query["slot"] for query in queries] == [1, 2, 3]
    assert [query["label"] for query in queries] == ["Yes", "No", "Yes"]  # P, N, P
    # 세 질문 모두 같은 이미지 3장을 같은 순서로 넣는다
    assert all(query["image_ids"] == [11, 21, 13] for query in queries)
    assert all(query["case_id"] == "group_01_dog_1_PNP" for query in queries)


def test_case_pairs_group_six_variants_per_pair_in_file_order():
    pairs = case_pairs(_make_cases())

    assert len(pairs) == 7 * 2 * 2  # 클래스 7개 x group 2개 x pair 2개
    assert [item["pair"] for item in pairs[:3]] == ["group_01_person_1", "group_01_person_2", "group_02_person_1"]
    for item in pairs:
        assert list(item["cases"]) == VARIANTS
        for variant, case in item["cases"].items():
            assert case["case_id"] == f"{item['pair']}_{variant}"


# -------------------------------------------------------- image_linked_classes


def test_classes_that_share_images_directly_or_through_another_class_form_one_group():
    # b와 c가 이미지 3을 같이 쓰고, c와 d가 이미지 5를 같이 쓴다. b, c, d는 한 덩어리다
    images = [("a", [1, 2]), ("b", [3, 4]), ("c", [3, 5]), ("d", [5, 6]), ("a", [2, 7]), ("e", [8])]
    cases = [{"object": name, "image_ids": image_ids} for name, image_ids in images]

    assert image_linked_classes(cases) == [{"a"}, {"b", "c", "d"}, {"e"}]  # 처음 나온 순서
