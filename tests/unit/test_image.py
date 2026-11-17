"""
src/mivlm/data/image.py 단위 테스트
pytest tests/unit/test_image.py -v
"""
from mivlm.config import COCO_IMAGE_DIR
from mivlm.data.image import image_path


def test_file_name_is_zero_padded_to_12_digits():
    assert image_path(139).name == "000000000139.jpg"
    assert image_path(581929).name == "000000581929.jpg"


def test_image_is_in_coco_image_dir():
    assert image_path(139).parent == COCO_IMAGE_DIR
