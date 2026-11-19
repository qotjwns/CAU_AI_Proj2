"""
실제 Qwen2.5-VL-3B로 pair 하나의 기록을 만들어 검사 (forward 6번 + 생성 6번)
pytest tests/integration/test_qwen_pair_record.py -v -s

pair는 group_05_cat_4를 쓴다. P2와 N1의 토큰 수가 같아서(391개) 위치가 통제되는 비교가 들어 있다.
"완전히 같다"는 기준은 이 기기(Mac, MPS, bf16)에서 관측한 것이다. 다른 기기에서 실패하면 출력된 값을 먼저 본다.
"""
import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from mivlm.analysis.pair_record import build_pair_record, observe_case, pair_summary
from mivlm.analysis.pair_store import save_pair_record
from mivlm.config import CASES_JSONL, COCO_IMAGE_DIR, load_models_config
from mivlm.data.cases import case_pairs, load_cases
from mivlm.models.inference import case_messages, infer, yes_no_token_ids
from mivlm.models.loading import load_model

MODEL_NAME = "qwen2.5-vl-3b"
PAIR = "group_05_cat_4"

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
def cases():
    (item,) = [item for item in case_pairs(load_cases()) if item["pair"] == PAIR]
    return item["cases"]  # {변형: 케이스}


@pytest.fixture(scope="module")
def record(loaded, cases):
    # observation 6개(각각 수백 MB)는 여기서만 쓰고 버린다. 테스트들은 작은 기록만 같이 쓴다
    model, processor = loaded
    observations = {variant: observe_case(model, processor, case) for variant, case in cases.items()}
    return build_pair_record(PAIR, observations)


def find(record: dict, image: str, a_variant: str, b_variant: str) -> dict:
    (comparison,) = [c for c in record["comparisons"]
                     if (c["image"], c["a"]["variant"], c["b"]["variant"]) == (image, a_variant, b_variant)]
    return comparison


def test_entry_0_is_identical_in_every_comparison(record):
    # [0]은 첫 층에 들어가기 전의 벡터라서, 같은 이미지는 어느 입력의 어느 자리에 있든 같다
    assert len(record["comparisons"]) == 12
    for comparison in record["comparisons"]:
        where = f"{comparison['image']}: {comparison['a']['variant']} vs {comparison['b']['variant']}"

        assert comparison["relative_l2"][0].item() == 0.0, where
        assert torch.allclose(comparison["token_cosine"][0], torch.ones(()), atol=1e-5), where


def test_same_prefix_reference_is_identical(record):
    # P2는 PPP와 PPN에서 앞부분이 완전히 같다. 측정값을 그대로 남기고, 완전히 같은지는 따로 적는다
    summary = pair_summary(record)
    print(f"\nsame_prefix: max relative L2 {summary['same_prefix_max_relative_l2']}, "
          f"identical {summary['same_prefix_identical']}")

    assert summary["same_prefix_max_relative_l2"] == 0.0
    assert summary["same_prefix_identical"] is True


def test_token_index_and_position_ids_are_recorded_independently(record):
    for comparison in record["comparisons"]:
        a, b = comparison["a"], comparison["b"]
        print(f"\n{comparison['family']:<13} {comparison['image']}: "
              f"{a['variant']} slot {a['slot']} tokens {a['token_span']} position {a['position_start']} | "
              f"{b['variant']} slot {b['slot']} tokens {b['token_span']} position {b['position_start']} | "
              f"same_token_index {comparison['same_token_index']}, same_position_ids {comparison['same_position_ids']}")

        assert comparison["same_token_index"] == (a["token_span"] == b["token_span"])
        # 같은 이미지는 격자가 같으므로, 첫 토큰의 position id가 같으면 나머지도 같다 (check_image_layout이 확인한 규칙)
        assert comparison["same_position_ids"] == (a["position_start"] == b["position_start"])

    # 자리 2의 P2와 N1이 둘 다 391토큰이라서, P3는 PPP와 PNP에서 토큰 인덱스도 position id도 같다
    controlled = find(record, "P3", "PPP", "PNP")
    assert (controlled["same_token_index"], controlled["same_position_ids"]) == (True, True)

    # 자리 1의 P1(299토큰)과 N1(391토큰)은 토큰 수는 다르지만 격자의 긴 변이 같다.
    # 그래서 P2는 PPP와 NPP에서 토큰 인덱스는 다르고 position id는 같다
    shifted = find(record, "P2", "PPP", "NPP")
    assert (shifted["same_token_index"], shifted["same_position_ids"]) == (False, True)

    # 자리가 바뀌면 둘 다 달라진다
    for comparison in record["comparisons"]:
        if comparison["family"] == "slot_change":
            assert (comparison["same_token_index"], comparison["same_position_ids"]) == (False, False)


def test_logits_match_infer_for_every_case(loaded, cases, record):
    # baseline과 같은 값이어야 한다: 같은 입력을 generate로 돌린 infer의 첫 생성 위치 logit
    model, processor = loaded
    for variant, case in cases.items():
        result = infer(model, processor, case_messages(case))
        saved = record["cases"][variant]
        print(f"\n{case['case_id']}: saved margin {saved['logit_margin']:+.3f}, "
              f"infer margin {result['logit_margin']:+.3f}")

        assert saved["yes_logit"] == result["yes_logit"], variant
        assert saved["no_logit"] == result["no_logit"], variant
        assert saved["logit_margin"] == result["logit_margin"], variant
        assert saved["input_token_count"] == result["input_token_count"], variant


def test_pre_answer_hidden_is_at_the_position_that_predicts_the_first_answer_token(loaded, cases, record):
    model, processor = loaded
    yes_id, no_id = yes_no_token_ids(processor.tokenizer)

    for variant, case in cases.items():
        saved = record["cases"][variant]
        # 입력은 assistant 머리말로 끝난다. 그 마지막 토큰 위치가 첫 답변 토큰을 예측하는 자리다
        prompt = processor.apply_chat_template(case_messages(case), tokenize=False, add_generation_prompt=True)
        assert prompt.endswith("<|im_start|>assistant\n"), variant
        assert saved["pre_answer_hidden"].shape == (37, 2048), variant

        # 마지막 항목(최종 norm이 적용된 벡터)을 lm_head에 넣으면 저장된 logit이 나와야 한다.
        # 벡터 하나만 따로 계산하므로 forward 때와 반올림이 다를 수 있어, bf16 간격(0.125) 두 칸까지 허용한다.
        # 다른 위치나 다른 층의 벡터라면 이 범위를 크게 벗어난다
        with torch.no_grad():
            logits = model.lm_head(saved["pre_answer_hidden"][-1].to(model.device, model.dtype)).float().cpu()
        print(f"\n{case['case_id']}: lm_head yes {logits[yes_id].item()} / no {logits[no_id].item()}, "
              f"saved yes {saved['yes_logit']} / no {saved['no_logit']}")

        assert logits[yes_id].item() == pytest.approx(saved["yes_logit"], abs=0.25), variant
        assert logits[no_id].item() == pytest.approx(saved["no_logit"], abs=0.25), variant


def test_saved_file_is_small_and_reads_back(tmp_path, record):
    saved = save_pair_record(tmp_path, record)
    path = tmp_path / saved["file"]
    megabytes = path.stat().st_size / 1e6
    print(f"\n{saved['file']}: {megabytes:.2f} MB")

    # 케이스 6개의 pre_answer_hidden 약 1.8 MB + 비교 12개의 토큰별 유사도 약 0.7 MB.
    # 전체 hidden state(케이스당 수백 MB)가 딸려 들어가면 이 크기를 크게 넘는다
    assert megabytes < 5

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert list(loaded) == ["pair", "cases", "comparisons"]
    assert list(loaded["cases"]) == list(record["cases"])
    for variant, case in record["cases"].items():
        assert loaded["cases"][variant]["token_spans"] == case["token_spans"]
        assert loaded["cases"][variant]["image_grid_thw"] == case["image_grid_thw"]
        assert torch.equal(loaded["cases"][variant]["pre_answer_hidden"], case["pre_answer_hidden"])
    for loaded_comparison, comparison in zip(loaded["comparisons"], record["comparisons"]):
        assert loaded_comparison["token_cosine"].dtype == torch.float32
        assert torch.equal(loaded_comparison["token_cosine"], comparison["token_cosine"])
        assert loaded_comparison["same_position_ids"] == comparison["same_position_ids"]
