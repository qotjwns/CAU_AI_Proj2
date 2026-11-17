"""
src/mivlm/data/image.py 단위 테스트
pytest tests/unit/test_image.py -v
"""
from pathlib import Path

from download_coco import COCO_ROOT
from mivlm.data.image import COCO_IMAGE_DIR, image_path


def test_file_name_is_zero_padded_to_12_digits():
    assert image_path(139).name == "000000000139.jpg"
    assert image_path(581929).name == "000000581929.jpg"


def test_image_dir_is_under_project_root():
    # test_image.py → unit → tests → 프로젝트 루트: image.py와 다른 경로로 루트를 계산해 대조한다
    project_root = Path(__file__).resolve().parents[2]

    assert COCO_IMAGE_DIR == project_root / "data" / "coco" / "val2017"
    assert image_path(139).parent == COCO_IMAGE_DIR


def test_matches_download_script_location():
    # 다운로드 스크립트가 이미지를 푸는 위치와 image_path가 읽는 위치가 같아야 한다
    assert COCO_IMAGE_DIR == COCO_ROOT / "val2017"
