"""
프로젝트 경로와 configs/ 설정 읽기
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # config.py → mivlm → src → 프로젝트 루트
MODELS_YAML = PROJECT_ROOT / "configs" / "models.yaml"


def load_models_config() -> dict[str, str]:
    """configs/models.yaml의 모델 목록: {이름: Hugging Face repo id}"""
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)["models"]
