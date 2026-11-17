"""
load_model의 dtype 지정 검사. 작은 모델(InternVL3.5-1B)을 써서 float32로도 가볍게 올라간다
pytest tests/integration/test_loading_dtype.py -v
"""
import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from mivlm.config import load_models_config
from mivlm.models.loading import load_model

MODEL_NAME = "internvl3.5-1b"

pytestmark = pytest.mark.skipif(
    not isinstance(try_to_load_from_cache(load_models_config()[MODEL_NAME], "config.json"), str),
    reason="모델이 없음: scripts/download_model.py internvl3.5-1b를 먼저 실행",
)


def test_dtype_can_be_set_to_float32_and_tf32_is_off():
    model, _ = load_model(MODEL_NAME, dtype=torch.float32)

    assert model.dtype == torch.float32
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
