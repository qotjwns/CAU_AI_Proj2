"""
src/mivlm/evaluation/decomposition.py 단위 테스트
pytest tests/unit/test_decomposition.py -v
"""
from mivlm.evaluation.decomposition import and_answers, combine_per_image, combine_single_image

DOG, CAT = 18, 17
# PPN 케이스: 자리 1, 2는 P(이미지 11, 12), 자리 3은 N(이미지 21)
CASE = {"case_id": "group_01_dog_1_PPN", "category_id": DOG, "object": "dog", "variant": "PPN",
        "image_ids": [11, 12, 21], "label": "No"}

# ----------------------------------------------------------------- and_answers


def test_and_is_yes_only_when_all_yes():
    assert and_answers(["Yes", "Yes", "Yes"]) == "Yes"
    assert and_answers(["Yes", "Yes", "No"]) == "No"
    assert and_answers(["No", "No", "No"]) == "No"


def test_and_with_unparsed_answers():
    assert and_answers(["Yes", None, "No"]) == "No"      # No가 하나라도 있으면 나머지와 상관없이 No
    assert and_answers(["Yes", None, "Yes"]) is None     # No는 없는데 판정 불가가 있으면 알 수 없다
    assert and_answers([None, None, None]) is None


# -------------------------------------------------------- combine_single_image


def _image_row(category_id, image_id, prediction, margin):
    return {"category_id": category_id, "image_id": image_id, "prediction": prediction, "logit_margin": margin}


def test_combine_single_image_looks_up_each_image_and_takes_min_margin():
    b_rows = [
        _image_row(DOG, 21, "No", -4.0),
        _image_row(DOG, 11, "Yes", 3.0),
        _image_row(DOG, 12, "Yes", 0.5),
    ]

    (row,) = combine_single_image([CASE], b_rows)

    assert row["image_predictions"] == ["Yes", "Yes", "No"]   # 케이스의 image_ids 순서 (11, 12, 21)
    assert row["image_margins"] == [3.0, 0.5, -4.0]
    assert row["prediction"] == "No"
    assert row["logit_margin"] == -4.0                        # 가장 작은 margin
    assert row["label"] == "No" and row["case_id"] == "group_01_dog_1_PPN"   # 케이스 필드는 그대로


def test_combine_single_image_uses_the_answer_for_the_case_class():
    # 이미지 21은 dog 질문에는 No, cat 질문에는 Yes다. dog 케이스는 dog의 답을 써야 한다
    b_rows = [
        _image_row(DOG, 11, "Yes", 3.0),
        _image_row(DOG, 12, "Yes", 2.0),
        _image_row(DOG, 21, "No", -4.0),
        _image_row(CAT, 21, "Yes", 5.0),
    ]

    (row,) = combine_single_image([CASE], b_rows)

    assert row["image_predictions"] == ["Yes", "Yes", "No"]


def test_false_yes_when_every_image_is_answered_yes():
    b_rows = [_image_row(DOG, 11, "Yes", 3.0), _image_row(DOG, 12, "Yes", 2.0), _image_row(DOG, 21, "Yes", 0.25)]

    (row,) = combine_single_image([CASE], b_rows)

    assert row["prediction"] == "Yes"      # 정답은 No이므로 False-Yes
    assert row["logit_margin"] == 0.25


# ----------------------------------------------------------- combine_per_image


def test_combine_per_image_orders_by_slot():
    # 결과 파일의 줄 순서가 섞여 있어도 자리 1, 2, 3 순서로 합친다
    c_rows = [
        {"case_id": CASE["case_id"], "slot": 3, "prediction": "No", "logit_margin": -1.5},
        {"case_id": CASE["case_id"], "slot": 1, "prediction": "Yes", "logit_margin": 2.0},
        {"case_id": CASE["case_id"], "slot": 2, "prediction": "No", "logit_margin": -0.25},
        {"case_id": "other_case", "slot": 1, "prediction": "Yes", "logit_margin": 9.0},
    ]

    (row,) = combine_per_image([CASE], c_rows)

    assert row["image_predictions"] == ["Yes", "No", "No"]
    assert row["prediction"] == "No"
    assert row["logit_margin"] == -1.5
