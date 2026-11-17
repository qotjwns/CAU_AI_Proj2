"""
추론: 메시지 구성 → 생성 → Yes/No 판정과 첫 토큰 logit margin
"""
from mivlm.data.image import image_path

QUESTION = "Is the object category '{object}' present in all 3 images? Please answer with Yes or No."
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
    return build_messages(case["image_ids"], QUESTION.format(object=case["object"]))


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


def infer(model, processor, messages: list[dict]) -> dict:
    """메시지 하나를 추론한다 (batch 1, greedy, 최대 8토큰)

    yes_logit, no_logit: 첫 생성 위치의 가공 전 logit에서 "Yes", "No" 토큰의 값
    logit_margin: yes_logit - no_logit. 양수면 첫 토큰으로 Yes가 No보다 그럴듯하다는 뜻
    """
    tokenizer = processor.tokenizer
    # "Yes", "No"가 토큰 하나가 아니면 여기서 에러가 난다
    (yes_id,) = tokenizer.encode("Yes", add_special_tokens=False)
    (no_id,) = tokenizer.encode("No", add_special_tokens=False)

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    out = model.generate(**inputs, **GENERATION_KWARGS, return_dict_in_generate=True, output_logits=True)
    first_logits = out.logits[0][0].float().cpu()  # 첫 생성 위치의 logit, shape (vocab 크기,)
    raw_response = tokenizer.decode(out.sequences[0, input_length:], skip_special_tokens=True)

    yes_logit = first_logits[yes_id].item()
    no_logit = first_logits[no_id].item()
    return {
        "raw_response": raw_response,
        "prediction": parse_yes_no(raw_response),
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "logit_margin": yes_logit - no_logit,
        "input_token_count": input_length,
        "image_token_count": (inputs["input_ids"] == processor.image_token_id).sum().item(),
    }
