"""
scripts/build_cases.py 단위 테스트
pytest tests/unit/test_build_cases.py -v
"""
from build_cases import make_cases

# p1, p2, p3 = 11, 12, 13 / n1, n2, n3 = 21, 22, 23
PAIR = {"group_id": "group_01_dog", "pair_id": 2, "category_id": 18, "object": "dog",
        "p": [11, 12, 13], "n": [21, 22, 23]}


def _cases_by_variant():
    return {case["variant"]: case for case in make_cases(PAIR)}


def test_six_variants_per_pair():
    assert len(make_cases(PAIR)) == 6
    assert sorted(_cases_by_variant()) == ["NNN", "NPP", "PNN", "PNP", "PPN", "PPP"]


def test_mode_a_image_order():
    cases = _cases_by_variant()

    assert cases["PPP"]["image_ids"] == [11, 12, 13]  # p1 p2 p3
    assert cases["PPN"]["image_ids"] == [11, 12, 21]  # p1 p2 n1
    assert cases["PNN"]["image_ids"] == [11, 22, 21]  # p1 n2 n1
    assert cases["NNN"]["image_ids"] == [23, 22, 21]  # n3 n2 n1


def test_mode_b_moves_only_n1():
    cases = _cases_by_variant()

    assert cases["NPP"]["image_ids"] == [21, 12, 13]  # n1 p2 p3
    assert cases["PNP"]["image_ids"] == [11, 21, 13]  # p1 n1 p3
    assert cases["PPN"]["image_ids"] == [11, 12, 21]  # p1 p2 n1


def test_variant_letters_match_images():
    # 모든 변형에서 variant 글자와 실제로 들어간 이미지의 P/N이 자리마다 일치해야 한다
    for case in make_cases(PAIR):
        letters = "".join("P" if image_id in PAIR["p"] else "N" for image_id in case["image_ids"])
        assert letters == case["variant"]


def test_only_ppp_is_yes():
    labels = {variant: case["label"] for variant, case in _cases_by_variant().items()}

    assert labels == {"PPP": "Yes", "PPN": "No", "PNN": "No", "NNN": "No", "NPP": "No", "PNP": "No"}


def test_mode_counts_and_shared_ppn():
    cases = make_cases(PAIR)

    assert sum("A" in case["modes"] for case in cases) == 4
    assert sum("B" in case["modes"] for case in cases) == 3
    assert _cases_by_variant()["PPN"]["modes"] == ["A", "B"]


def test_case_id_and_copied_fields():
    case = _cases_by_variant()["PNN"]

    assert case["case_id"] == "group_01_dog_2_PNN"
    assert (case["group_id"], case["pair_id"], case["category_id"], case["object"]) == ("group_01_dog", 2, 18, "dog")
