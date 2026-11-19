"""
실제 Qwen2.5-VL로 hidden state와 position id 꺼내기 검사 (이미지 3장 케이스 하나, 기본 3B)
pytest tests/integration/test_qwen_hidden_states.py -v -s
MIVLM_TEST_MODEL=qwen2.5-vl-7b pytest tests/integration/test_qwen_hidden_states.py -v -s     # 다른 모델로
MIVLM_TEST_DTYPE=float32 pytest tests/integration/test_qwen_hidden_states.py -v -s           # 다른 dtype으로
"""
import os

import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from mivlm.analysis.hidden_states import extract_hidden_states, position_ids
from mivlm.config import CASES_JSONL, COCO_IMAGE_DIR, load_models_config
from mivlm.data.cases import load_cases
from mivlm.models.inference import case_messages, image_token_spans, infer, prepare_inputs, yes_no_token_ids
from mivlm.models.loading import load_model

# 다른 모델로 돌리려면 환경 변수로 이름을 준다. 예: MIVLM_TEST_MODEL=qwen2.5-vl-7b pytest <이 파일>
MODEL_NAME = os.environ.get("MIVLM_TEST_MODEL", "qwen2.5-vl-3b")
# 다른 dtype으로 돌리려면: MIVLM_TEST_DTYPE=float32 pytest <이 파일>. 주지 않으면 device의 기본값 (CUDA, MPS는 bfloat16)
DTYPE = os.environ.get("MIVLM_TEST_DTYPE")
CASE_ID = "group_01_person_1_PPN"

pytestmark = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(load_models_config()[MODEL_NAME], "config.json"), str)
    or not CASES_JSONL.exists()
    or not COCO_IMAGE_DIR.exists(),
    reason=f"모델, 케이스 파일 또는 COCO 이미지가 없음: scripts/download_model.py {MODEL_NAME}, "
           "scripts/download_coco.py, build_cases.py를 먼저 실행",
)


@pytest.fixture(scope="module")
def loaded():
    return load_model(MODEL_NAME, getattr(torch, DTYPE) if DTYPE else None)


@pytest.fixture(scope="module")
def messages():
    (case,) = [case for case in load_cases() if case["case_id"] == CASE_ID]
    return case_messages(case)


@pytest.fixture(scope="module")
def inputs(loaded, messages):
    # extract_hidden_states와 position_ids는 device로 옮긴 입력을 받는다
    model, processor = loaded
    return prepare_inputs(processor, messages).to(model.device)


@pytest.fixture(scope="module")
def extracted(loaded, inputs):
    model, _ = loaded
    return extract_hidden_states(model, inputs)


def test_shape_is_all_layers_all_tokens(loaded, inputs, extracted):
    # 모델 설정에 적힌 층 수와 벡터 크기에 맞춘다. 첫 층에 들어가기 전의 벡터가 하나 더 있다 (3B는 36 + 1 = 37개, 크기 2048)
    model, _ = loaded
    text_config = model.config.text_config
    hidden = extracted["hidden_states"]
    print(f"\nhidden_states={tuple(hidden.shape)}, {hidden.dtype}, {hidden.device}")

    assert hidden.shape == (text_config.num_hidden_layers + 1, inputs["input_ids"].shape[1], text_config.hidden_size)
    assert hidden.dtype == torch.float32
    assert hidden.device.type == "cpu"
    assert torch.isfinite(hidden).all()


def test_logits_match_infer(loaded, messages, extracted):
    # 같은 입력을 generate로 돌린 infer의 값과 같아야 한다: 꺼낸 hidden state가 baseline과 같은 계산에서 나왔다는 근거
    model, processor = loaded
    yes_id, no_id = yes_no_token_ids(processor.tokenizer)
    yes_logit, no_logit = extracted["logits"][yes_id].item(), extracted["logits"][no_id].item()

    result = infer(model, processor, messages)
    print(f"\nforward: yes {yes_logit} / no {no_logit}, infer: yes {result['yes_logit']} / no {result['no_logit']}")

    assert yes_logit == result["yes_logit"]
    assert no_logit == result["no_logit"]


def test_layer_0_differs_from_token_embedding_exactly_at_image_tokens(loaded, inputs, extracted):
    # [0]은 첫 층에 들어가기 전의 벡터다: 글자 위치는 embedding 그대로이고, 이미지 위치는 vision encoder의 출력으로 바뀌어 있다
    model, processor = loaded
    input_ids = inputs["input_ids"].to(model.device)
    with torch.no_grad():
        embedding = model.get_input_embeddings()(input_ids)[0].float().cpu()  # 모든 위치를 글자 토큰으로 봤을 때의 벡터
    is_image = (input_ids[0] == processor.image_token_id).cpu()
    differs = (extracted["hidden_states"][0] != embedding).any(dim=1)  # 위치마다: embedding과 다른가
    print(f"\nimage tokens {is_image.sum().item()}, positions that differ from the embedding {differs.sum().item()}")

    assert torch.equal(differs, is_image)


def test_position_ids_match_what_the_forward_uses(loaded, inputs):
    # forward가 언어 모델에 실제로 넘기는 position_ids를 hook으로 잡아서 position_ids()의 값과 비교한다.
    # position_ids()를 먼저 부르고 forward를 하므로, 미리 불러도 forward에 영향이 없다는 것도 같이 확인된다
    model, _ = loaded
    expected = position_ids(model, inputs)

    captured = {}

    def capture(module, args, kwargs):  # 언어 모델의 forward가 불리기 직전에 실행된다
        captured["position_ids"] = kwargs["position_ids"]

    handle = model.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        extract_hidden_states(model, inputs)
    finally:
        handle.remove()  # hook을 남겨 두면 이후의 모든 forward에서 계속 실행된다

    used = captured["position_ids"]  # (3, 1, 토큰 수)
    print(f"\nposition ids used by the forward: {tuple(used.shape)}, {used.dtype}")

    assert expected.shape == (3, inputs["input_ids"].shape[1])
    assert torch.equal(used[:, 0].cpu(), expected)


def test_position_ids_follow_the_image_grid(loaded, inputs):
    # 글자 토큰: 세 축이 같다. 이미지 토큰: 시간 축은 이미지 전체가 같고, 토큰은 행 우선 순서(왼쪽 → 오른쪽, 위 → 아래)다.
    # 이미지 뒤의 토큰은 토큰 수가 아니라 max(행 수, 열 수)만큼 건너뛴다
    model, processor = loaded
    ids = position_ids(model, inputs)
    input_ids = inputs["input_ids"][0].cpu()
    is_image = input_ids == processor.image_token_id
    merge = processor.image_processor.merge_size

    assert torch.equal(ids[0, ~is_image], ids[1, ~is_image])
    assert torch.equal(ids[0, ~is_image], ids[2, ~is_image])

    spans = image_token_spans(input_ids, processor.image_token_id)
    for slot, ((start, end), (_, grid_h, grid_w)) in enumerate(zip(spans, inputs["image_grid_thw"].tolist()), start=1):
        rows, columns = grid_h // merge, grid_w // merge
        first = ids[0, start].item()  # 이 이미지의 첫 토큰이 받은 번호
        print(f"\nslot {slot}: tokens {start}-{end - 1}, {rows} rows x {columns} columns, "
              f"position ids start at {first}, next token gets {ids[0, end].item()}")

        assert ids[0, start:end].tolist() == [first] * (rows * columns), slot
        assert ids[1, start:end].tolist() == (first + torch.arange(rows).repeat_interleave(columns)).tolist(), slot
        assert ids[2, start:end].tolist() == (first + torch.arange(columns).repeat(rows)).tolist(), slot
        assert ids[0, end].item() == first + max(rows, columns), slot
