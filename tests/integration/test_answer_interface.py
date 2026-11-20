"""
모델의 입력과 출력이 이 연구의 존재 판단 과제와 같은 뜻인지 실제 모델로 검사 (dev 풀의 파일럿 케이스만 쓴다)
MIVLM_TEST_MODEL=internvl3.5-4b pytest tests/integration/test_answer_interface.py -v -s    # InternVL 파일럿
pytest tests/integration/test_answer_interface.py -v -s                                   # 기본 모델(Qwen 3B)

확인하는 것
    1. "Yes"와 "No"가 각각 토큰 하나이고, 그 id를 디코딩하면 정확히 "Yes", "No"다 (앞 공백이 붙은 토큰이 아니다)
    2. 렌더한 프롬프트에 질문 문구가 그대로 있고 thinking 지시(<think>)가 없다
    3. 이미지 3장의 토큰 묶음이 메시지의 이미지 순서대로 놓인다
       (한 장씩 넣었을 때의 토큰 수가 3장 입력의 묶음 길이와 순서대로 같고, 순서를 뒤집으면 길이도 뒤집힌다)
    4. 실제 첫 생성 토큰이 "Yes"나 "No"이고, 파싱한 답과 같고, 가공 전 margin의 부호와 같다. 같은 입력을 다시 넣으면
       같은 logit이 나온다
confirm, eval 풀은 읽지 않는다. 모델, dev 풀, COCO 이미지가 없으면 건너뛴다.
"""
import pytest
from conftest import MODEL_NAME

from mivlm.config import CASES_DIR, COCO_IMAGE_DIR
from mivlm.data.cases import load_cases, pilot_cases
from mivlm.models.inference import (
    ALL_IMAGES_QUESTION, build_messages, case_messages, image_token_spans, infer, prepare_inputs, yes_no_token_ids,
)

pytestmark = pytest.mark.skipif(not (CASES_DIR / "dev" / "cases.jsonl").exists() or not COCO_IMAGE_DIR.exists(),
                                reason="dev 풀이나 COCO 이미지가 없음")


@pytest.fixture(scope="module")
def pilot():
    """dev 풀의 파일럿 케이스 (PPP, PPN, NPP, PNP가 클래스 순서로 돌아간다)"""
    return pilot_cases(load_cases("dev"), "dev")


def _image_token_count(processor, image_ids: list[int], question: str) -> list[int]:
    inputs = prepare_inputs(processor, build_messages(image_ids, question))
    return [end - start for start, end in image_token_spans(inputs["input_ids"][0], processor.image_token_id)]


def test_yes_and_no_are_single_tokens_that_decode_back(loaded):
    _, processor = loaded
    yes_id, no_id = yes_no_token_ids(processor.tokenizer)

    assert processor.tokenizer.decode([yes_id]) == "Yes"
    assert processor.tokenizer.decode([no_id]) == "No"


def test_the_prompt_has_the_question_and_no_thinking_prefix(loaded, pilot):
    _, processor = loaded
    case = pilot[0]

    text = processor.apply_chat_template(case_messages(case), tokenize=False, add_generation_prompt=True)
    print(f"\n[{MODEL_NAME}] 렌더한 프롬프트 (이미지 자리 표시 포함):\n{text}")

    assert ALL_IMAGES_QUESTION.format(object=case["object"]) in text
    assert "<think>" not in text


def test_image_tokens_follow_the_image_order(loaded, pilot):
    _, processor = loaded
    case = pilot[0]
    question = ALL_IMAGES_QUESTION.format(object=case["object"])
    alone = [_image_token_count(processor, [image_id], question)[0] for image_id in case["image_ids"]]

    together = _image_token_count(processor, case["image_ids"], question)
    reversed_order = _image_token_count(processor, list(reversed(case["image_ids"])), question)
    print(f"\n[{MODEL_NAME}] 이미지별 토큰 수 {alone}, 3장 입력의 묶음 {together}")

    assert together == alone  # 묶음이 3개이고, 길이가 이미지 순서대로다
    assert reversed_order == list(reversed(alone))
    tokens_per_tile = getattr(processor, "image_seq_length", None)  # InternVL: 타일 하나가 토큰 몇 개인가
    if tokens_per_tile:
        assert all(count % tokens_per_tile == 0 for count in alone)


def test_the_first_token_parse_and_margin_agree_and_repeat(loaded, pilot):
    model, processor = loaded
    yes_id, no_id = yes_no_token_ids(processor.tokenizer)

    for case in pilot[:2]:  # PPP와 PPN
        first = infer(model, processor, case_messages(case))
        again = infer(model, processor, case_messages(case))
        print(f"\n[{MODEL_NAME}] {case['case_id']}: {first}")

        assert first["first_token"] in ("Yes", "No") and first["first_token_id"] in (yes_id, no_id)
        assert first["prediction"] == first["first_token"]
        if first["logit_margin"] != 0:
            assert (first["prediction"] == "Yes") == (first["logit_margin"] > 0)
        assert (again["yes_logit"], again["no_logit"]) == (first["yes_logit"], first["no_logit"])
