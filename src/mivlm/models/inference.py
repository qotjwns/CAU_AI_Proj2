"""
추론: 메시지 구성 → 모델 입력 만들기 → 생성 → Yes/No 판정과 첫 토큰 logit margin
"""
import torch

from mivlm.data.image import image_path

# 질문 문구는 실험의 핵심 조건이다. 바꾸면 이미 받은 결과와 다른 실험이 된다
# A: 이미지 3장을 넣고 3장 모두에 있는지 묻는다
ALL_IMAGES_QUESTION = "Is the object category '{object}' present in all 3 images? Please answer with Yes or No."
# B: 이미지 1장만 넣고 묻는다
SINGLE_IMAGE_QUESTION = "Is the object category '{object}' present in this image? Please answer with Yes or No."
# C: 이미지 3장을 넣고 그중 한 자리를 지목해서 묻는다
PER_IMAGE_QUESTION = "Is the object category '{object}' present in the {ordinal} image? Please answer with Yes or No."
ORDINALS = ["first", "second", "third"]
GENERATION_KWARGS = {"do_sample": False, "num_beams": 1, "max_new_tokens": 8}


def build_messages(image_ids: list[int], question: str) -> list[dict]:
    """user 턴 하나를 만든다: 이미지들 먼저, 질문 마지막

    system 메시지는 넣지 않는다. chat template이 기본 문구를 자동으로 넣는다.
    이미지는 절대 경로 문자열로 넘기고, 여는 일은 processor가 한다.
    """
    content = [{"type": "image", "image": str(image_path(image_id))} for image_id in image_ids]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def case_messages(case: dict) -> list[dict]:
    """cases.jsonl의 케이스 하나를 메시지로 만든다: 이미지 3장(image_ids 순서 그대로) → 질문"""
    return build_messages(case["image_ids"], ALL_IMAGES_QUESTION.format(object=case["object"]))


def single_image_messages(item: dict) -> list[dict]:
    """이미지 1장만 넣고 그 이미지에 대해 묻는 메시지 (분해 실험 B)"""
    return build_messages([item["image_id"]], SINGLE_IMAGE_QUESTION.format(object=item["object"]))


def per_image_messages(query: dict) -> list[dict]:
    """이미지 3장을 그대로 넣고, 그중 한 자리(query["slot"], 1부터 셈)의 이미지에 대해 묻는 메시지"""
    question = PER_IMAGE_QUESTION.format(object=query["object"], ordinal=ORDINALS[query["slot"] - 1])
    return build_messages(query["image_ids"], question)


def prepare_inputs(processor, messages: list[dict]):
    """메시지를 모델 입력으로 바꾼다 (CPU 텐서). 추론과 내부 값 분석이 같은 입력을 쓰도록 한 곳에서 만든다

    Qwen2.5-VL 기준으로 들어 있는 것:
    input_ids, attention_mask: shape (1, 토큰 수)
    pixel_values: shape (모든 이미지의 패치 수 합, 1176)
    image_grid_thw: shape (이미지 수, 3). 이미지마다 (1, 세로 패치 수, 가로 패치 수)
    add_generation_prompt=True라서 마지막 토큰은 assistant 머리말의 끝이고, 모델은 그 다음 토큰부터 답을 쓴다.
    """
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )


def image_token_spans(input_ids: torch.Tensor, image_token_id: int) -> list[tuple[int, int]]:
    """이미지마다 그 토큰이 놓인 구간 (start, end)를 입력 순서대로 돌려준다. end는 포함하지 않는다

    input_ids: 입력 하나의 토큰 id, shape (토큰 수,)
    한 이미지의 토큰은 연속으로 놓이고 이미지 사이에는 다른 토큰이 있으므로, 연속된 묶음 하나가 이미지 하나다.
    층 하나의 hidden state가 hidden (토큰 수, 벡터 크기)일 때 hidden[start:end]가 그 이미지의 표현이다.
    """
    positions = (input_ids == image_token_id).nonzero().flatten().tolist()
    spans = []
    for position in positions:
        if spans and spans[-1][1] == position:  # 바로 앞 구간에 이어지는 토큰
            spans[-1] = (spans[-1][0], position + 1)
        else:  # 새 이미지의 첫 토큰
            spans.append((position, position + 1))
    return spans


def parse_yes_no(text: str) -> str | None:
    """생성된 텍스트를 "Yes" / "No" / None으로 판정한다

    앞뒤 공백과 끝의 . ! ? 를 지우고 대소문자를 무시해서 yes, no와 같은지 본다.
    """
    answer = text.strip().rstrip(".!?").lower()
    if answer == "yes":
        return "Yes"
    if answer == "no":
        return "No"
    return None


def yes_no_token_ids(tokenizer) -> tuple[int, int]:
    """"Yes"와 "No"의 토큰 id를 (yes_id, no_id)로 돌려준다. 둘 중 하나라도 토큰 하나가 아니면 에러가 난다"""
    (yes_id,) = tokenizer.encode("Yes", add_special_tokens=False)
    (no_id,) = tokenizer.encode("No", add_special_tokens=False)
    return yes_id, no_id


def first_generated_token(tokenizer, sequences: torch.Tensor, input_length: int) -> tuple[int, str]:
    """실제로 생성된 첫 토큰: (token id, 그 토큰 하나를 디코딩한 문자열)

    sequences: generate가 돌려준 토큰 id, shape (1, 입력 길이 + 생성 길이). 입력 바로 뒤의 토큰이 첫 생성 토큰이다.
    특수 토큰(예: <think>)과 앞 공백을 그대로 남겨 디코딩한다. 첫 토큰이 정말 "Yes"나 "No"인지 확인하는 데 쓴다.
    """
    token_id = int(sequences[0, input_length])
    return token_id, tokenizer.decode([token_id])


def infer(model, processor, messages: list[dict]) -> dict:
    """메시지 하나를 추론한다 (batch 1, greedy, 최대 8토큰)

    yes_logit, no_logit: 첫 생성 위치의 가공 전 logit에서 "Yes", "No" 토큰의 값
    logit_margin: yes_logit - no_logit. 양수면 첫 토큰으로 Yes가 No보다 그럴듯하다는 뜻
    first_token_id, first_token: 실제로 생성된 첫 토큰 (first_generated_token). 생성은 repetition penalty 같은
        후처리를 거친 점수에서 고르므로, 가공 전 logit의 margin 부호와 다를 수 있다. 두 값을 따로 남긴다
    """
    tokenizer = processor.tokenizer
    yes_id, no_id = yes_no_token_ids(tokenizer)

    inputs = prepare_inputs(processor, messages).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    out = model.generate(**inputs, **GENERATION_KWARGS, return_dict_in_generate=True, output_logits=True)
    first_logits = out.logits[0][0].float().cpu()  # 첫 생성 위치의 logit, shape (vocab 크기,)
    raw_response = tokenizer.decode(out.sequences[0, input_length:], skip_special_tokens=True)

    yes_logit = first_logits[yes_id].item()
    no_logit = first_logits[no_id].item()
    first_token_id, first_token = first_generated_token(tokenizer, out.sequences, input_length)
    return {
        "raw_response": raw_response,
        "prediction": parse_yes_no(raw_response),
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "logit_margin": yes_logit - no_logit,
        "first_token_id": first_token_id,
        "first_token": first_token,
        "input_token_count": input_length,
        "image_token_count": (inputs["input_ids"] == processor.image_token_id).sum().item(),
    }
