"""
실제 Qwen2.5-VL-3B에서 "앞부분이 같으면 그 앞부분의 hidden state도 같다"를 검사 (forward 9번)
pytest tests/integration/test_qwen_hidden_state_causality.py -v -s

모델은 각 토큰에서 앞쪽 토큰만 보므로, 두 입력이 어느 위치까지 같으면 그 위치까지의 벡터는 뒤가 달라도 같아야 한다.
이것이 성립하면 이미지 토큰 구간, 추출 코드, "이미지 위치의 벡터는 질문을 모른다"는 전제가 함께 확인된다.
"앞부분이 다르면 벡터가 다르다"는 구조가 보장하는 것이 아니라 측정할 대상이므로 여기서 판정하지 않는다.

이미지 3장짜리 입력끼리는 완전히 같은지(torch.equal)로 판정한다.
이미지 1장짜리 입력과의 비교만 범위로 판정한다. 이유는 test_single_image_input_matches_slot_1_within_rounding_error에 적었다.

group_01_person pair 1의 이미지 P1, P2, P3, N1을 쓴다. 모든 비교의 차이를 출력하므로 -s를 붙여서 본다.
"""
import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from mivlm.analysis.hidden_states import extract_hidden_states
from mivlm.config import CASES_JSONL, COCO_IMAGE_DIR, load_models_config
from mivlm.data.cases import load_cases
from mivlm.models.inference import (
    case_messages, image_token_spans, per_image_messages, prepare_inputs, single_image_messages,
)
from mivlm.models.loading import load_model

MODEL_NAME = "qwen2.5-vl-3b"
PAIR = ("group_01_person", 1)  # (group_id, pair_id)

pytestmark = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(load_models_config()[MODEL_NAME], "config.json"), str)
    or not CASES_JSONL.exists()
    or not COCO_IMAGE_DIR.exists(),
    reason="모델, 케이스 파일 또는 COCO 이미지가 없음: scripts/download_model.py qwen2.5-vl-3b, "
           "scripts/download_coco.py, build_cases.py를 먼저 실행",
)


@pytest.fixture(scope="module")
def loaded():
    return load_model(MODEL_NAME)


@pytest.fixture(scope="module")
def runs(loaded):
    """입력 9개를 한 번씩 forward 한 결과: {이름: {"image_ids", "input_ids", "spans", "hidden"}}

    PPP, PPN, PNP, NPP: 이미지 3장 + "세 장 모두에 있는가" (실험 A)
    PPN again: PPN과 같은 입력을 한 번 더
    C: PPN과 같은 이미지 + "첫 번째 이미지에 있는가" (실험 C)
    P1 alone, P2 alone, N1 alone: 이미지 1장 + "이 이미지에 있는가" (실험 B)
    hidden: shape (37, 토큰 수, 2048)
    """
    model, processor = loaded
    cases = {case["variant"]: case for case in load_cases() if (case["group_id"], case["pair_id"]) == PAIR}
    ppn = cases["PPN"]
    p1, p2, n1 = ppn["image_ids"]

    planned = [(variant, cases[variant]["image_ids"], case_messages(cases[variant]))
               for variant in ["PPP", "PPN", "PNP", "NPP"]]
    planned.append(("PPN again", ppn["image_ids"], case_messages(ppn)))
    planned.append(("C", ppn["image_ids"],
                    per_image_messages({"object": ppn["object"], "image_ids": ppn["image_ids"], "slot": 1})))
    for name, image_id in [("P1 alone", p1), ("P2 alone", p2), ("N1 alone", n1)]:
        planned.append((name, [image_id], single_image_messages({"image_id": image_id, "object": ppn["object"]})))

    results = {}
    for name, image_ids, messages in planned:
        inputs = prepare_inputs(processor, messages)
        input_ids = inputs["input_ids"][0]  # extract_hidden_states가 inputs를 device로 옮기기 전의 CPU 텐서
        results[name] = {
            "image_ids": image_ids,
            "input_ids": input_ids,
            "spans": image_token_spans(input_ids, processor.image_token_id),
            "hidden": extract_hidden_states(model, inputs)["hidden_states"],
        }
    return results


def report(label: str, a: torch.Tensor, b: torch.Tensor) -> bool:
    """두 벡터 묶음의 차이를 출력하고, 완전히 같은지 돌려준다

    a, b: shape (층 수, 토큰 수, 벡터 크기). 층마다 계산해서 가장 나쁜 층의 값을 적는다.
    max abs diff: 값 하나하나의 차이 중 최댓값
    relative L2: 차이의 크기 / 원래 크기. cosine이 못 보는 벡터 크기의 차이까지 잡는다
    min cosine: 토큰별 cosine similarity 중 최솟값
    """
    difference = a - b
    max_abs = difference.abs().amax(dim=(1, 2))
    relative_l2 = difference.flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1)
    min_cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1).amin(dim=1)
    print(f"\n{label}: max abs diff {max_abs.max().item():.3g}, "
          f"relative L2 {relative_l2.max().item():.3g} (layer {relative_l2.argmax().item()}), "
          f"min cosine {min_cosine.min().item():.6f} (layer {min_cosine.argmin().item()})")
    return torch.equal(a, b)


def shared_prefix(runs: dict, first: str, second: str, n_images: int) -> tuple[str, torch.Tensor, torch.Tensor]:
    """두 입력이 n_images번째 이미지까지 실제로 같은지 확인하고, 거기까지의 벡터를 돌려준다: (설명, a, b)

    a, b: shape (층 수, 그 구간의 토큰 수, 벡터 크기)
    """
    a, b = runs[first], runs[second]
    end = a["spans"][n_images - 1][1]

    # 전제: 그 구간에 놓인 이미지와 토큰이 같다 (이미지 토큰의 id는 이미지가 달라도 같으므로 둘 다 본다)
    assert a["image_ids"][:n_images] == b["image_ids"][:n_images], f"{first} vs {second}"
    assert torch.equal(a["input_ids"][:end], b["input_ids"][:end]), f"{first} vs {second}"

    return f"{first} vs {second}, tokens 0-{end - 1}", a["hidden"][:, :end], b["hidden"][:, :end]


def test_same_input_twice_gives_the_same_hidden_states(runs):
    # 기준선: 같은 입력도 다르게 나온다면 나머지 비교는 의미가 없다
    assert report("PPN vs PPN again, all tokens", runs["PPN"]["hidden"], runs["PPN again"]["hidden"])


def test_slot_1_does_not_depend_on_what_comes_after(runs):
    # 자리 1의 이미지 P1까지는 PPP, PPN, PNP가 모두 같다
    different = [other for other in ["PPN", "PNP"] if not report(*shared_prefix(runs, "PPP", other, n_images=1))]

    assert not different, f"PPP와 완전히 같지 않은 입력: {different}"


def test_slots_1_and_2_do_not_depend_on_slot_3(runs):
    # PPP와 PPN은 세 번째 이미지만 다르다
    assert report(*shared_prefix(runs, "PPP", "PPN", n_images=2))


def test_images_do_not_depend_on_the_question(runs):
    # PPN과 C는 이미지 3장이 같고 질문만 다르다. 질문은 이미지 뒤에 온다
    assert report(*shared_prefix(runs, "PPN", "C", n_images=3))


def test_single_image_input_matches_slot_1_within_rounding_error(runs):
    # P1만 넣은 입력(실험 B)도 P1까지는 PPP와 같은 토큰이지만, Mac(MPS, bf16)에서는 벡터가 완전히 같지 않다.
    # 진단으로 확인한 사실:
    #   - 차이는 입력 전체가 짧을 때만 생긴다. 같은 이미지 1장에 질문만 길게 붙이면 PPP와 완전히 같아진다
    #   - 두 입력의 position id는 같고, 같은 입력을 두 번 돌리면 완전히 같다
    #   - 처음 몇 층은 완전히 같고, 중간의 한 층에서 아주 작게 시작해 층을 지나며 서서히 커진다
    # 그래서 내용이 아니라 입력 길이에 따라 계산이 다르게 처리되어 생기는 반올림 오차로 본다 (어느 연산인지는 찾지 않았다).
    # 아래 범위는 그 오차보다 여유를 둔 값이고, 크게 어긋나는 것을 잡는 용도다.
    # 오차가 "앞부분이 달라서 생기는 실제 차이"보다 작은지는 이 테스트가 판정하지 않는다.
    # 다른 device(CUDA)에서도 같은 현상이 있는지는 확인하지 않았다.
    label, a, b = shared_prefix(runs, "PPP", "P1 alone", n_images=1)
    report(label, a, b)
    relative_l2 = (a - b).flatten(1).norm(dim=1) / a.flatten(1).norm(dim=1)  # 층마다
    mean_cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean(dim=1)  # 층마다 토큰별 값의 평균

    assert relative_l2.max().item() < 0.05
    assert mean_cosine.min().item() > 0.999


def test_layer_0_of_an_image_is_the_same_wherever_it_is(runs):
    # [0]은 첫 층에 들어가기 전의 벡터이므로, 같은 이미지는 어느 자리, 어느 입력에 있든 같아야 한다
    # (vision encoder가 이미지를 서로 독립으로 처리한다는 뜻이기도 하다)
    comparisons = [  # ((입력, 자리), (입력, 자리)). 자리는 1부터 센다
        (("PPN", 1), ("P1 alone", 1)), (("PPN", 2), ("P2 alone", 1)), (("PPN", 3), ("N1 alone", 1)),  # 각 자리와 그 이미지만
        (("NPP", 1), ("PPN", 3)), (("PNP", 2), ("PPN", 3)),  # N1이 자리 1, 2, 3에 있을 때
        (("PPP", 3), ("NPP", 3)),  # P3: 자리는 같고 앞의 이미지가 다를 때
    ]
    different = []
    for (first, first_slot), (second, second_slot) in comparisons:
        a, b = runs[first], runs[second]
        label = f"layer 0, {first} slot {first_slot} vs {second} slot {second_slot}"
        assert a["image_ids"][first_slot - 1] == b["image_ids"][second_slot - 1], label  # 전제: 같은 이미지다

        a_start, a_end = a["spans"][first_slot - 1]
        b_start, b_end = b["spans"][second_slot - 1]
        if not report(label, a["hidden"][:1, a_start:a_end], b["hidden"][:1, b_start:b_end]):
            different.append(label)

    assert not different, f"완전히 같지 않은 비교: {different}"
