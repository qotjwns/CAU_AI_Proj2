"""
src/mivlm/evaluation/decomposition.py 단위 테스트
pytest tests/unit/test_decomposition.py -v
"""
from mivlm.evaluation.decomposition import (
    a_false_yes_answers, and_answers, combine_per_image, combine_single_image, images_all_correct, main_metrics,
    one_n_outcomes,
)

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


# ---------------------------------------------------------- images_all_correct


def test_images_all_correct_when_each_slot_matches_the_variant():
    assert images_all_correct({"variant": "PPN", "image_predictions": ["Yes", "Yes", "No"]})
    assert images_all_correct({"variant": "NNN", "image_predictions": ["No", "No", "No"]})


def test_and_can_be_right_while_an_image_answer_is_wrong():
    # PPN: N 자리를 No라고 했으므로 AND는 정답(No)이지만, 자리 1의 P를 No라고 놓쳤다
    row = {"variant": "PPN", "image_predictions": ["No", "Yes", "No"]}

    assert and_answers(row["image_predictions"]) == "No"
    assert not images_all_correct(row)


def test_images_all_correct_compares_by_slot_not_by_count():
    # Yes 2개, No 1개로 개수는 PNP와 같지만 자리가 다르다
    assert not images_all_correct({"variant": "PNP", "image_predictions": ["Yes", "Yes", "No"]})


def test_images_all_correct_is_false_with_an_unparsed_answer():
    assert not images_all_correct({"variant": "PPP", "image_predictions": ["Yes", None, "Yes"]})


# ------------------------------------- main_metrics, a_false_yes_answers, one_n_outcomes


def _case_row(variant, prediction, case_id="case", image_predictions=None):
    """케이스 단위의 행에서 세는 데 쓰이는 필드만 담은 행. 정답은 PPP만 Yes다"""
    return {"case_id": case_id, "variant": variant, "label": "Yes" if variant == "PPP" else "No",
            "prediction": prediction, "logit_margin": 1.0 if prediction == "Yes" else -1.0,
            "image_predictions": image_predictions}


def test_main_metrics_uses_all_cases_then_one_n_variants_then_ppp():
    rows = [
        _case_row("PPP", "Yes"), _case_row("PPP", "No"),    # PPP 정확도 1/2
        _case_row("PPN", "Yes"), _case_row("PNP", "No"), _case_row("NPP", "No"), _case_row("NPP", "No"),  # False-Yes 1/4
        _case_row("NNN", "Yes"), _case_row("PNN", "No"),    # N이 1장이 아니라서 False-Yes rate에는 들어가지 않는다
    ]

    # 맞힌 케이스: PPP 1 + N 1장 3 + PNN 1 = 5개
    assert main_metrics(rows) == {"accuracy": 5 / 8, "one_n_false_yes_rate": 1 / 4, "ppp_accuracy": 1 / 2}


def test_a_false_yes_answers_counts_the_b_and_c_answers_of_those_cases_only():
    case_rows = {
        "A": [_case_row("PPN", "Yes", "c1"), _case_row("PNP", "Yes", "c2"), _case_row("NPP", "Yes", "c3"),
              _case_row("PPN", "No", "c4"),      # A가 맞혔으므로 세지 않는다
              _case_row("PPP", "Yes", "c5")],    # 정답이 Yes이므로 False-Yes가 아니다
        "B": [_case_row("PPN", "No", "c1"), _case_row("PNP", "No", "c2"), _case_row("NPP", "No", "c3"),
              _case_row("PPN", "No", "c4"), _case_row("PPP", "Yes", "c5")],
        "C": [_case_row("PPN", "No", "c1"), _case_row("PNP", "Yes", "c2"), _case_row("NPP", None, "c3"),
              _case_row("PPN", "No", "c4"), _case_row("PPP", "Yes", "c5")],
    }

    assert a_false_yes_answers(case_rows) == {("No", "No"): 1, ("No", "Yes"): 1, ("No", None): 1}


def test_one_n_outcomes_separates_the_image_errors_that_and_hides():
    rows = [
        _case_row("PPN", "No", image_predictions=["Yes", "Yes", "No"]),    # 합친 답도 이미지별 답도 맞음
        _case_row("PNP", "No", image_predictions=["No", "No", "Yes"]),     # 합친 답은 맞지만 자리 1의 P를 놓침
        _case_row("NPP", "Yes", image_predictions=["Yes", "Yes", "Yes"]),  # False-Yes
        _case_row("NPP", None, image_predictions=["Yes", None, "Yes"]),    # 판정 불가: 어느 쪽에도 세지 않는다
        _case_row("PPP", "No", image_predictions=["Yes", "Yes", "No"]),    # N이 1장인 변형이 아니다
    ]

    assert one_n_outcomes(rows) == {"n": 4, "answered_no": 2, "all_images_correct": 1, "false_yes": 1}
