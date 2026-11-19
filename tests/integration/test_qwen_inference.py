"""
실제 Qwen2.5-VL로 이미지 1장 추론 검사 (기본 3B)
pytest tests/integration/test_qwen_inference.py -v -s
"""
import pytest

from mivlm.data.image import image_path
from mivlm.models.inference import SINGLE_IMAGE_QUESTION, build_messages, infer

CAT_IMAGE_ID = 39769  # 고양이 두 마리가 소파에 누워 있는 640x480 사진

# 모델(loaded)은 conftest.py가 로딩한다. 모델이 없으면 거기서 건너뛴다
pytestmark = pytest.mark.skipif(
    not image_path(CAT_IMAGE_ID).exists(),
    reason="COCO 이미지가 없음: scripts/download_coco.py를 먼저 실행",
)


def _ask(loaded, object_name):
    model, processor = loaded
    result = infer(model, processor, build_messages([CAT_IMAGE_ID], SINGLE_IMAGE_QUESTION.format(object=object_name)))
    print(f"\n{object_name}: {result}")
    return result


@pytest.fixture(scope="module")
def cat_result(loaded):
    return _ask(loaded, "cat")


@pytest.fixture(scope="module")
def airplane_result(loaded):
    return _ask(loaded, "airplane")


def test_default_system_prompt_is_inserted(loaded):
    _, processor = loaded
    text = processor.apply_chat_template(
        build_messages([CAT_IMAGE_ID], "QUESTION"), tokenize=False, add_generation_prompt=True
    )
    print(f"\n{text!r}")

    assert text == (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>QUESTION<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_token_counts(loaded, cat_result):
    # 이미지: 640x480 → 644x476 → 14px 패치 격자 34x46 = 1564개 → 2x2로 합쳐 토큰 391개
    # 그 외: system 블록 11 + user 머리 3 + vision 시작/끝 2 + 턴 끝 2 + assistant 머리 3 = 21, 여기에 질문 토큰
    _, processor = loaded
    question_tokens = len(processor.tokenizer.encode(SINGLE_IMAGE_QUESTION.format(object="cat"), add_special_tokens=False))

    assert cat_result["image_token_count"] == 391
    assert cat_result["input_token_count"] == 391 + 21 + question_tokens


def test_margin_is_yes_minus_no(cat_result):
    assert cat_result["logit_margin"] == cat_result["yes_logit"] - cat_result["no_logit"]


# 아래 둘은 논리 검증이 아니라 모델 동작에 대한 예상이다.
# 고양이 사진에서 틀리면 이미지가 모델에 제대로 들어가지 않았을 가능성이 높다.


def test_present_object_is_answered_yes(cat_result):
    assert cat_result["prediction"] == "Yes"
    assert cat_result["logit_margin"] > 0


def test_absent_object_is_answered_no(airplane_result):
    assert airplane_result["prediction"] == "No"
    assert airplane_result["logit_margin"] < 0
