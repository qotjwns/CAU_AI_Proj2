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

CASES_DIR = PROJECT_ROOT / "data" / "cases"  # 기존 탐색 데이터(풀 이름 없음)의 케이스 파일 폴더
POOLS_JSON = CASES_DIR / "pools.json"
PAIRS_JSON = CASES_DIR / "pairs.json"
CASES_JSONL = CASES_DIR / "cases.jsonl"
# 새 이미지 풀 (scripts/build_new_pools.py, docs/preregistration.md 2.2절). 파일은 CASES_DIR / <풀 이름> / 아래에 있다
POOL_NAMES = ("dev", "confirm", "eval")
# 추론하려면 --unseal로 한 번 더 이름을 적어야 하는 풀. 실수를 막는 장치일 뿐이고, 언제 열어도 되는지는 사전 등록의 규칙을 따른다
# (confirm은 Stage 2 등록 뒤, eval은 최종 평가 단계)
SEALED_POOLS = ("confirm", "eval")

PREDICTIONS_DIR = PROJECT_ROOT / "outputs" / "predictions"  # <실험>/<실행 이름>/{meta.json, results.jsonl}
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"  # <실험>/<실행 이름>/<그림>.png
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"  # <실험>/<표>.csv


def pool_dir(pool: str | None) -> Path:
    """케이스 파일(cases.jsonl, pools.json, pairs.json)의 폴더

    pool이 None이면 기존 탐색 데이터의 CASES_DIR, POOL_NAMES의 이름이면 CASES_DIR / pool이다. 그 밖의 값은 ValueError.
    경로는 이 함수에서만 정한다. CASES_DIR는 부를 때마다 이 모듈에서 읽으므로 테스트에서 바꿔 끼울 수 있다.
    """
    if pool is None:
        return CASES_DIR
    if pool not in POOL_NAMES:
        raise ValueError(f"알 수 없는 풀 이름: {pool!r}. 쓸 수 있는 이름은 {POOL_NAMES} (기존 탐색 데이터는 None)")
    return CASES_DIR / pool


def load_models_config() -> dict[str, str]:
    """configs/models.yaml의 모델 목록: {이름: Hugging Face repo id}"""
    with open(MODELS_YAML) as f:
        return yaml.safe_load(f)["models"]
