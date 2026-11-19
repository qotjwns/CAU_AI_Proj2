"""
통합 테스트가 같이 쓰는 것: 테스트할 모델과 dtype, 그리고 한 번만 로딩하는 모델

다른 모델로 돌리려면: MIVLM_TEST_MODEL=qwen2.5-vl-7b pytest tests/integration/<파일>
다른 dtype으로 돌리려면: MIVLM_TEST_DTYPE=float32 pytest tests/integration/<파일>
    주지 않으면 device의 기본값이다 (CUDA, MPS는 bfloat16)

pytest는 이 파일을 같은 폴더의 테스트보다 먼저 읽고, 여기 정의한 fixture를 그 테스트들에 자동으로 제공한다.
MODEL_NAME과 DTYPE이 필요한 테스트 파일은 from conftest import MODEL_NAME처럼 가져다 쓴다.
"""
import os

import pytest
import torch
from huggingface_hub import try_to_load_from_cache

from mivlm.config import load_models_config
from mivlm.models.loading import load_model

MODEL_NAME = os.environ.get("MIVLM_TEST_MODEL", "qwen2.5-vl-3b")
DTYPE = os.environ.get("MIVLM_TEST_DTYPE")


@pytest.fixture(scope="session")
def loaded():
    """(model, processor). pytest를 한 번 실행하는 동안 한 번만 로딩해서 모든 테스트 파일이 같이 쓴다

    파일마다 따로 로딩하면 통합 테스트 전체를 돌릴 때 같은 모델을 여러 번 올리게 된다.
    모델을 받아 두지 않았으면 이 fixture를 쓰는 테스트는 건너뛴다.
    """
    if not isinstance(try_to_load_from_cache(load_models_config()[MODEL_NAME], "config.json"), str):
        pytest.skip(f"모델이 없음: scripts/download_model.py {MODEL_NAME}을 먼저 실행")
    return load_model(MODEL_NAME, getattr(torch, DTYPE) if DTYPE else None)
