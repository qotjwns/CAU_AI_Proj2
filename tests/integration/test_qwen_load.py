"""
실제 Qwen2.5-VL 로딩과 이미지 전처리 검사 (기본 3B)
pytest tests/integration/test_qwen_load.py -v -s
"""
import pytest
import torch
from conftest import DTYPE

from mivlm.data.image import image_path
from mivlm.models.loading import pick_device

GRAY_IMAGE_ID = 209222  # 흑백(L 모드) 640x427. 이전 STEP 1에서도 쓰인 이미지

# 모델(loaded)은 conftest.py가 로딩한다. 모델이 없으면 거기서 건너뛴다
pytestmark = pytest.mark.skipif(
    not image_path(GRAY_IMAGE_ID).exists(),
    reason="COCO 이미지가 없음: scripts/download_coco.py를 먼저 실행",
)


def test_device_and_dtype(loaded):
    # dtype을 따로 주지 않았으면(MIVLM_TEST_DTYPE 없음) device의 기본값으로 로딩돼야 한다
    model, _ = loaded
    device, default_dtype = pick_device()
    print(f"\ndevice={model.device}, dtype={model.dtype}")

    assert model.device.type == device
    assert model.dtype == (getattr(torch, DTYPE) if DTYPE else default_dtype)


def test_eval_mode(loaded):
    model, _ = loaded

    assert not model.training


def test_default_attention_is_sdpa(loaded):
    model, _ = loaded

    assert model.config._attn_implementation == "sdpa"


def test_image_processor_is_torchvision_based(loaded):
    _, processor = loaded

    assert type(processor.image_processor).__name__ == "Qwen2VLImageProcessor"


def test_grayscale_image_becomes_three_channels(loaded):
    # 640x427 → 28의 배수로 맞추면 644x420 → 14px 패치 격자 30x46 = 1380개 → 2x2로 합쳐 토큰 345개
    # 패치 하나 = 3(RGB) x 2(시간) x 14 x 14 = 1176개 값. RGB 변환이 안 됐다면 392가 된다
    _, processor = loaded
    messages = [{"role": "user", "content": [
        {"type": "image", "image": str(image_path(GRAY_IMAGE_ID))},
        {"type": "text", "text": "Describe this image."},
    ]}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    print(f"\npixel_values={tuple(inputs['pixel_values'].shape)}, grid={inputs['image_grid_thw'].tolist()}, "
          f"input_ids={tuple(inputs['input_ids'].shape)}")

    assert tuple(inputs["pixel_values"].shape) == (1380, 1176)
    assert inputs["image_grid_thw"].tolist() == [[1, 30, 46]]
    assert (inputs["input_ids"] == processor.image_token_id).sum().item() == 345
