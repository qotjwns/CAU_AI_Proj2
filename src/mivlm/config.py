"""
프로젝트 경로와 configs/ 설정 읽기
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # config.py → mivlm → src → 프로젝트 루트
MODELS_YAML = PROJECT_ROOT / "configs" / "models.yaml"

COCO_ROOT = PROJECT_ROOT / "data" / "coco"
COCO_IMAGE_DIR = COCO_ROOT / "val2017"
INSTANCES_JSON = COCO_ROOT / "annotations" / "instances_valrem.json"

CASES_DIR = PROJECT_ROOT / "data" / "cases"
POOLS_JSON = CASES_DIR / "pools.json"
PAIRS_JSON = CASES_DIR / "pairs.json"
CASES_JSONL = CASES_DIR / "cases.jsonl"


def load_models_config() -> dict[str, str]:
    """configs/models.yaml의 모델 목록: {이름: Hugging Face repo id}"""
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)["models"]
