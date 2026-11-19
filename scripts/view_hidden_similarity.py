"""
pair 하나에서 같은 이미지의 hidden state가 조건(자리, 앞의 이미지)에 따라 얼마나 달라지는지 층별 표로 출력
python scripts/view_hidden_similarity.py group_01_person_1

판정은 하지 않고 값만 본다. 결과는 터미널에만 출력하고 저장하지 않는다.
이미지 이름: PPP의 세 장이 P1, P2, P3이고 PPN의 세 번째가 N1이다. PNP는 [P1, N1, P3], NPP는 [N1, P2, P3]이다.
"""
import argparse

import torch

from mivlm.analysis.hidden_states import compare_hidden_states, extract_hidden_states
from mivlm.data.cases import load_cases
from mivlm.evaluation.report import print_layer_table
from mivlm.models.inference import case_messages, image_token_spans, prepare_inputs, single_image_messages
from mivlm.models.loading import load_model

MODEL_NAME = "qwen2.5-vl-3b"

# (설명, (입력, 자리), (입력, 자리)). 자리는 1부터 센다
SAME_IMAGE = [  # 같은 이미지를 두 조건에서: 토큰끼리 비교할 수 있다
    ("P1, 자리 1: PPP와 P1만        (달라진 것 없음. 계산 오차의 크기)", ("PPP", 1), ("P1 alone", 1)),
    ("P2, 자리 2: PPP와 NPP         (자리 1의 이미지가 다름)", ("PPP", 2), ("NPP", 2)),
    ("P3, 자리 3: PPP와 PNP         (자리 2의 이미지가 다름)", ("PPP", 3), ("PNP", 3)),
    ("P3, 자리 3: PPP와 NPP         (자리 1의 이미지가 다름)", ("PPP", 3), ("NPP", 3)),
    ("N1: 자리 1(NPP)과 자리 2(PNP) (자리와 앞의 이미지가 다름)", ("NPP", 1), ("PNP", 2)),
    ("N1: 자리 1(NPP)과 자리 3(PPN) (자리와 앞의 이미지가 다름)", ("NPP", 1), ("PPN", 3)),
    ("N1: 자리 2(PNP)와 자리 3(PPN) (자리와 앞의 이미지가 다름)", ("PNP", 2), ("PPN", 3)),
]
DIFFERENT_IMAGES = [  # 서로 다른 이미지: 토큰 수가 달라서 평균끼리만 비교한다
    ("자리 3: P3(PPP)와 N1(PPN)     (이미지 자체가 다름)", ("PPP", 3), ("PPN", 3)),
    ("자리 1: P1(PPP)과 N1(NPP)     (이미지 자체가 다름)", ("PPP", 1), ("NPP", 1)),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="pair 하나의 hidden state 유사도를 층별 표로 출력")
    parser.add_argument("pair", help="group_id와 pair_id를 이은 이름. 예: group_01_person_1")
    args = parser.parse_args()

    cases_by_id = {case["case_id"]: case for case in load_cases()}
    cases = {variant: cases_by_id[f"{args.pair}_{variant}"] for variant in ["PPP", "PPN", "PNP", "NPP"]}
    p1 = cases["PPP"]["image_ids"][0]
    planned = {variant: (case["image_ids"], case_messages(case)) for variant, case in cases.items()}
    planned["P1 alone"] = ([p1], single_image_messages({"image_id": p1, "object": cases["PPP"]["object"]}))

    model, processor = load_model(MODEL_NAME)
    (yes_id,) = processor.tokenizer.encode("Yes", add_special_tokens=False)
    (no_id,) = processor.tokenizer.encode("No", add_special_tokens=False)
    images = {}  # (입력 이름, 자리) -> (image_id, 그 이미지의 벡터들 shape (층 수, 토큰 수, 벡터 크기))
    for name, (image_ids, messages) in planned.items():
        inputs = prepare_inputs(processor, messages)
        spans = image_token_spans(inputs["input_ids"][0], processor.image_token_id)
        extracted = extract_hidden_states(model, inputs)
        hidden = extracted["hidden_states"]
        margin = (extracted["logits"][yes_id] - extracted["logits"][no_id]).item()  # 양수면 Yes
        for slot, (image_id, (start, end)) in enumerate(zip(image_ids, spans), start=1):
            images[name, slot] = (image_id, hidden[:, start:end])
        print(f"{name}: images {image_ids}, tokens per image {[end - start for start, end in spans]}, "
              f"margin {margin:+.3f}")

    # 비교마다 번호를 붙인다. 표의 열 이름으로 쓴다
    print(f"\n[비교 목록] {args.pair}, {MODEL_NAME}")
    metrics = {}  # 비교 번호 -> {지표 이름: 층마다의 값}
    for number, (description, first, second) in enumerate(SAME_IMAGE + DIFFERENT_IMAGES, start=1):
        (first_id, a), (second_id, b) = images[first], images[second]
        print(f"{number:>3}  {description}")
        if number <= len(SAME_IMAGE):
            assert first_id == second_id, f"{description}: 같은 이미지가 아니다 ({first_id}, {second_id})"
            metrics[number] = compare_hidden_states(a, b)
        else:
            assert first_id != second_id, f"{description}: 다른 이미지가 아니다 ({first_id})"
            means = torch.nn.functional.cosine_similarity(a.mean(dim=1), b.mean(dim=1), dim=-1)
            metrics[number] = {"cosine_of_means": means}

    same = range(1, len(SAME_IMAGE) + 1)
    print_layer_table("1. 토큰별 유사도: 토큰끼리 cosine을 구한 뒤 평균",
                {number: metrics[number]["token_cosine"].mean(dim=1) for number in same}, ".5f")
    print_layer_table("2. 평균끼리의 유사도: 토큰을 먼저 평균 낸 벡터끼리의 cosine",
                {number: values["cosine_of_means"] for number, values in metrics.items()}, ".5f")
    print_layer_table("3. Relative L2: 차이의 크기 / 원래 크기",
                {number: metrics[number]["relative_l2"] for number in same}, ".4f")


if __name__ == "__main__":
    main()
