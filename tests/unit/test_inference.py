"""
src/mivlm/models/inference.py 단위 테스트 (모델은 로딩하지 않는다)
pytest tests/unit/test_inference.py -v
"""
from mivlm.data.image import image_path
from mivlm.models.inference import build_messages, case_messages, parse_yes_no

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
