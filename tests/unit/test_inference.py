"""
src/mivlm/models/inference.py 단위 테스트 (모델은 로딩하지 않는다)
pytest tests/unit/test_inference.py -v
"""
import pytest
import torch

from mivlm.data.image import image_path
from mivlm.models.inference import (
    build_messages, case_messages, image_token_spans, parse_yes_no, per_image_messages, single_image_messages,
    yes_no_token_ids,
)

PAD = 9  # 테스트용 가짜 이미지 토큰 id

# ---------------------------------------------------------------- parse_yes_no


def test_parse_yes():
    assert parse_yes_no("Yes") == "Yes"
    assert parse_yes_no("yes.") == "Yes"
    assert parse_yes_no(" YES! ") == "Yes"


def test_parse_no():
    assert parse_yes_no("No") == "No"
    assert parse_yes_no("No.") == "No"
    assert parse_yes_no("\nno?") == "No"


def test_parse_anything_else_is_none():
    assert parse_yes_no("Yes, it is") is None
    assert parse_yes_no("The image shows") is None
    assert parse_yes_no("Yes No") is None
    assert parse_yes_no("") is None


# -------------------------------------------------------------- build_messages


def test_single_user_turn_without_system():
    messages = build_messages([139], "question?")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_images_come_first_then_question():
    content = build_messages([139, 285, 632], "question?")[0]["content"]

    assert [part["type"] for part in content] == ["image", "image", "image", "text"]
    assert content[-1]["text"] == "question?"


def test_images_keep_order_as_absolute_path_strings():
    content = build_messages([285, 139], "question?")[0]["content"]

    assert content[0]["image"] == str(image_path(285))
    assert content[1]["image"] == str(image_path(139))
    assert content[0]["image"].startswith("/")


# --------------------------------------------------------------- case_messages


def test_case_becomes_three_images_then_question():
    case = {"case_id": "group_01_dining_table_1_PNN", "object": "dining table", "image_ids": [11, 22, 21]}

    content = case_messages(case)[0]["content"]

    assert [part.get("image") for part in content[:3]] == [str(image_path(i)) for i in [11, 22, 21]]
    assert content[3]["text"] == (
        "Is the object category 'dining table' present in all 3 images? Please answer with Yes or No."
    )


# ---------------------------------------------------------- per_image_messages


def test_per_image_question_names_the_slot():
    query = {"object": "dining table", "slot": 2, "image_ids": [11, 21, 13]}

    content = per_image_messages(query)[0]["content"]

    assert [part.get("image") for part in content[:3]] == [str(image_path(i)) for i in [11, 21, 13]]
    assert content[3]["text"] == (
        "Is the object category 'dining table' present in the second image? Please answer with Yes or No."
    )


def test_per_image_question_ordinals():
    texts = [per_image_messages({"object": "dog", "slot": slot, "image_ids": [1, 2, 3]})[0]["content"][3]["text"]
             for slot in [1, 2, 3]]

    assert ["first" in texts[0], "second" in texts[1], "third" in texts[2]] == [True, True, True]


# ------------------------------------------------------- single_image_messages


def test_single_image_message_has_one_image_and_the_b_question():
    content = single_image_messages({"object": "dining table", "image_id": 139})[0]["content"]

    assert [part["type"] for part in content] == ["image", "text"]
    assert content[0]["image"] == str(image_path(139))
    assert content[1]["text"] == "Is the object category 'dining table' present in this image? Please answer with Yes or No."


# ----------------------------------------------------------- image_token_spans


def test_spans_one_per_image_in_input_order():
    #                 위치:   0  1  2    3    4  5  6    7    8    9  10
    input_ids = torch.tensor([1, 2, PAD, PAD, 3, 4, PAD, PAD, PAD, 5, 6])

    assert image_token_spans(input_ids, PAD) == [(2, 4), (6, 9)]


def test_span_slices_contain_only_image_tokens():
    input_ids = torch.tensor([1, 2, PAD, PAD, 3, 4, PAD, PAD, PAD, 5, 6])

    spans = image_token_spans(input_ids, PAD)

    # end를 포함하지 않으므로 input_ids[start:end]가 그 이미지의 토큰 전부이고, end - start가 토큰 수다
    assert [input_ids[start:end].tolist() for start, end in spans] == [[PAD, PAD], [PAD, PAD, PAD]]
    assert [end - start for start, end in spans] == [2, 3]


def test_spans_empty_without_image_tokens():
    assert image_token_spans(torch.tensor([1, 2, 3]), PAD) == []


def test_spans_at_both_ends_of_the_input():
    assert image_token_spans(torch.tensor([PAD, PAD, 1, PAD]), PAD) == [(0, 2), (3, 4)]


# ----------------------------------------------------------- yes_no_token_ids


class FakeTokenizer:
    """글자마다 정해 둔 토큰 id 목록을 돌려주는 가짜 tokenizer"""

    def __init__(self, ids_by_text: dict[str, list[int]]):
        self.ids_by_text = ids_by_text

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return self.ids_by_text[text]


def test_yes_no_token_ids_come_in_yes_no_order():
    assert yes_no_token_ids(FakeTokenizer({"Yes": [7], "No": [3]})) == (7, 3)


def test_yes_or_no_split_into_several_tokens_is_an_error():
    with pytest.raises(ValueError):
        yes_no_token_ids(FakeTokenizer({"Yes": [7, 8], "No": [3]}))
