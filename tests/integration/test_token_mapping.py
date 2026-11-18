"""
실제 Qwen2.5-VL-3B processor로 image_token_spans의 구간이 그 자리의 이미지를 가리키는지 검사
pytest tests/integration/test_token_mapping.py -v

모델 가중치는 로딩하지 않고 processor만 쓴다.
대상: 클래스마다 첫 pair(group_01의 pair 1)의 NPP, PNP, PPN. 같은 N 이미지가 자리 1, 2, 3으로 옮겨 간다.
구간 길이만으로는 어느 이미지인지 알 수 없으므로(크기가 같으면 토큰 수도 같다) 내용, 길이, 위치를 따로 맞춰 본다.
"""
import pytest
import torch
from huggingface_hub import try_to_load_from_cache
from transformers import AutoProcessor

from mivlm.config import CASES_JSONL, COCO_IMAGE_DIR, load_models_config
from mivlm.data.cases import load_cases
from mivlm.models.inference import case_messages, image_token_spans, prepare_inputs, single_image_messages

MODEL_NAME = "qwen2.5-vl-3b"
N_MOVES = ["NPP", "PNP", "PPN"]  # N이 자리 1 → 2 → 3으로 옮겨 가는 순서. 목록에서의 순번이 곧 N의 자리다
FIRST_IMAGE_TOKEN = 15  # 앞에 system 블록 11 + user 머리 3 + <|vision_start|> 1
TOKENS_BETWEEN_IMAGES = 2  # <|vision_end|>, <|vision_start|>

pytestmark = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(load_models_config()[MODEL_NAME], "config.json"), str)
    or not CASES_JSONL.exists()
    or not COCO_IMAGE_DIR.exists(),
    reason="모델, 케이스 파일 또는 COCO 이미지가 없음: scripts/download_model.py qwen2.5-vl-3b, "
           "scripts/download_coco.py, build_cases.py를 먼저 실행",
)


@pytest.fixture(scope="module")
def processor():
    return AutoProcessor.from_pretrained(load_models_config()[MODEL_NAME], local_files_only=True)


@pytest.fixture(scope="module")
def prepared(processor):
    """대상 케이스마다 processor를 한 번만 돌린 결과. 테스트들이 같이 쓴다

    input_ids: shape (토큰 수,)
    spans: 이미지 3장의 토큰 구간 (start, end). end는 포함하지 않는다
    grid: 이미지마다 (1, 세로 패치 수, 가로 패치 수)
    patches: pixel_values를 이미지별로 나눈 것. 한 장의 shape는 (세로 패치 수 x 가로 패치 수, 1176)
    """
    items = []
    for case in load_cases():
        if case["group_id"].startswith("group_01_") and case["pair_id"] == 1 and case["variant"] in N_MOVES:
            inputs = prepare_inputs(processor, case_messages(case))
            grid = inputs["image_grid_thw"].tolist()
            items.append({
                "case": case,
                "input_ids": inputs["input_ids"][0],
                "spans": image_token_spans(inputs["input_ids"][0], processor.image_token_id),
                "grid": grid,
                "patches": inputs["pixel_values"].split([h * w for _, h, w in grid]),
            })
    return items


@pytest.fixture(scope="module")
def alone(processor, prepared):
    """대상 케이스에 나오는 이미지를 한 장씩 혼자 넣었을 때의 결과: {image_id: {"patches", "n_tokens"}}"""
    result = {}
    for item in prepared:
        for image_id in item["case"]["image_ids"]:
            if image_id not in result:
                messages = single_image_messages({"image_id": image_id, "object": item["case"]["object"]})
                inputs = prepare_inputs(processor, messages)
                ((start, end),) = image_token_spans(inputs["input_ids"][0], processor.image_token_id)
                result[image_id] = {"patches": inputs["pixel_values"], "n_tokens": end - start}
    return result


def test_three_spans_cover_all_image_tokens(processor, prepared):
    # 대상이 비어 있으면 아래 검사들이 전부 그냥 통과하므로 개수부터 본다: 클래스 20개 x 변형 3개
    assert len(prepared) == 60

    for item in prepared:
        case_id = item["case"]["case_id"]
        n_image_tokens = (item["input_ids"] == processor.image_token_id).sum().item()

        assert len(item["spans"]) == 3, case_id
        assert sum(end - start for start, end in item["spans"]) == n_image_tokens, case_id


def test_each_slot_holds_its_image(prepared, alone):
    # 내용: 자리 k의 패치가, image_ids의 k번째 이미지를 혼자 넣었을 때의 패치와 완전히 같아야 한다
    for item in prepared:
        for slot, image_id in enumerate(item["case"]["image_ids"]):
            where = f"{item['case']['case_id']} 자리 {slot + 1} 이미지 {image_id}"
            in_case, by_itself = item["patches"][slot], alone[image_id]["patches"]

            assert in_case.shape == by_itself.shape, where
            assert torch.equal(in_case, by_itself), f"{where}: 최대 차이 {(in_case - by_itself).abs().max().item()}"


def test_span_length_matches_grid(processor, prepared):
    # 길이: 패치를 2x2씩 합쳐 토큰 하나가 되므로 토큰 수 = 세로 x 가로 // 4
    merge = processor.image_processor.merge_size
    for item in prepared:
        for slot, ((start, end), (_, grid_h, grid_w)) in enumerate(zip(item["spans"], item["grid"])):
            assert end - start == grid_h * grid_w // merge**2, f"{item['case']['case_id']} 자리 {slot + 1}"


def test_n_image_keeps_token_count_as_it_moves(prepared, alone):
    # 같은 N 이미지가 NPP, PNP, PPN에서 자리만 바뀐다: 토큰 수는 그대로이고 구간은 뒤로 밀린다
    by_object = {}
    for item in prepared:
        by_object.setdefault(item["case"]["object"], {})[item["case"]["variant"]] = item

    for name, items in by_object.items():
        n_image_ids = [items[variant]["case"]["image_ids"][slot] for slot, variant in enumerate(N_MOVES)]
        n_spans = [items[variant]["spans"][slot] for slot, variant in enumerate(N_MOVES)]
        starts = [start for start, _ in n_spans]

        assert len(set(n_image_ids)) == 1, name
        assert [end - start for start, end in n_spans] == [alone[n_image_ids[0]]["n_tokens"]] * 3, name
        assert starts[0] < starts[1] < starts[2], name


def test_span_positions_follow_from_lengths(processor, prepared):
    # 위치: 첫 이미지는 15번째 토큰부터, 다음 이미지는 앞 이미지가 끝나고 2토큰 뒤부터
    vision_start = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    for item in prepared:
        case_id, input_ids, spans = item["case"]["case_id"], item["input_ids"], item["spans"]

        assert spans[0][0] == FIRST_IMAGE_TOKEN, case_id
        for (_, previous_end), (start, _) in zip(spans, spans[1:]):
            assert start == previous_end + TOKENS_BETWEEN_IMAGES, case_id
        for start, end in spans:
            assert input_ids[start - 1].item() == vision_start, case_id
            assert input_ids[end].item() == vision_end, case_id


def test_known_spans(prepared):
    # 기준점: scripts/view_case_input.py로 실제 확인한 값. cases.jsonl을 다시 만들어 이 케이스의 이미지가 바뀌면 달라진다
    (item,) = [item for item in prepared if item["case"]["case_id"] == "group_01_person_1_PPN"]

    assert item["spans"] == [(15, 383), (385, 845), (847, 1192)]
