"""
실제 Qwen2.5-VL-3B로 hidden state 꺼내기 검사 (이미지 3장 케이스 하나)
pytest tests/integration/test_qwen_hidden_states.py -v -s
"""
import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from mivlm.analysis.hidden_states import extract_hidden_states
from mivlm.config import CASES_JSONL, COCO_IMAGE_DIR, load_models_config
from mivlm.data.cases import load_cases
from mivlm.models.inference import case_messages, infer, prepare_inputs
from mivlm.models.loading import load_model

MODEL_NAME = "qwen2.5-vl-3b"
CASE_ID = "group_01_person_1_PPN"

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
def messages():
    (case,) = [case for case in load_cases() if case["case_id"] == CASE_ID]
    return case_messages(case)


@pytest.fixture(scope="module")
def inputs(loaded, messages):
    _, processor = loaded
    return prepare_inputs(processor, messages)


@pytest.fixture(scope="module")
def extracted(loaded, inputs):
    model, _ = loaded
    return extract_hidden_states(model, inputs)


def test_shape_is_all_layers_all_tokens(inputs, extracted):
    # 3B: 층 36개에 첫 층에 들어가기 전의 벡터를 더해 37개, 벡터 크기 2048
    hidden = extracted["hidden_states"]
    print(f"\nhidden_states={tuple(hidden.shape)}, {hidden.dtype}, {hidden.device}")

    assert hidden.shape == (37, inputs["input_ids"].shape[1], 2048)
    assert hidden.dtype == torch.float32
    assert hidden.device.type == "cpu"
    assert torch.isfinite(hidden).all()


def test_logits_match_infer(loaded, messages, extracted):
    # 같은 입력을 generate로 돌린 infer의 값과 같아야 한다: 꺼낸 hidden state가 baseline과 같은 계산에서 나왔다는 근거
    model, processor = loaded
    (yes_id,) = processor.tokenizer.encode("Yes", add_special_tokens=False)
    (no_id,) = processor.tokenizer.encode("No", add_special_tokens=False)
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
