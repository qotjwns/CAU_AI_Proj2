"""
모델과 processor 로딩 (모델 공통)
모델 종류(Qwen, InternVL 등)는 Auto 클래스가 모델 config.json의 model_type을 보고 고른다
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from mivlm.config import load_models_config


def pick_device() -> tuple[str, torch.dtype]:
    """device와 dtype 선택: CUDA(bf16, 지원 안 하면 fp16) → MPS(bf16) → CPU(fp32)"""
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    return "cpu", torch.float32


def load_model(name: str, dtype: torch.dtype | None = None):
    """configs/models.yaml의 이름으로 모델과 processor를 로딩한다. 반환: (model, processor)

    dtype을 주지 않으면 device에 맞는 기본값(pick_device)을 쓴다.
    local_files_only=True: 캐시에 받아 둔 버전만 쓴다 (scripts/download_model.py로 먼저 받기).
    HF 저장소가 업데이트돼도 새 버전으로 바뀌지 않는다.
    """
    repo_id = load_models_config()[name]
    device, default_dtype = pick_device()
    if dtype is None:
        dtype = default_dtype
    # TF32는 float32 연산을 빠르게 하려고 정밀도를 낮추는 기능이다 (합성곱에는 기본으로 켜져 있다).
    # float32 실행이 정확한 float32가 되도록 끈다. bf16 실행에는 영향이 없다
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForImageTextToText.from_pretrained(repo_id, dtype=dtype, local_files_only=True).to(device)
    processor = AutoProcessor.from_pretrained(repo_id, local_files_only=True)
    return model, processor
