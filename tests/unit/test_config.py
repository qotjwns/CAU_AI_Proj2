"""
src/mivlm/config.py 단위 테스트
pytest tests/unit/test_config.py -v
"""
from pathlib import Path

from mivlm.config import MODELS_YAML, PROJECT_ROOT, load_models_config


def test_project_root():
    # test_config.py → unit → tests → 프로젝트 루트: config.py와 다른 경로로 계산해 대조한다
    assert PROJECT_ROOT == Path(__file__).resolve().parents[2]
    assert MODELS_YAML == PROJECT_ROOT / "configs" / "models.yaml"


def test_models_config_maps_name_to_repo_id():
    models = load_models_config()

    assert models["qwen2.5-vl-3b"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert all(isinstance(repo_id, str) for repo_id in models.values())
