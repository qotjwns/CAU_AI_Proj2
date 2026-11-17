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


def load_model(name: str):
    """configs/models.yaml의 이름으로 모델과 processor를 로딩한다. 반환: (model, processor)

    local_files_only=True: 캐시에 받아 둔 버전만 쓴다 (scripts/download_model.py로 먼저 받기).
    HF 저장소가 업데이트돼도 새 버전으로 바뀌지 않는다.
    """
    repo_id = load_models_config()[name]
    device, dtype = pick_device()
    model = AutoModelForImageTextToText.from_pretrained(repo_id, dtype=dtype, local_files_only=True).to(device)
    processor = AutoProcessor.from_pretrained(repo_id, local_files_only=True)
    return model, processor
