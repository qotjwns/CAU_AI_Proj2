"""
COCO val2017 이미지 경로
이미지 경로 생성
"""
from pathlib import Path

from mivlm.config import COCO_IMAGE_DIR


def image_path(image_id: int) -> Path:
    """COCO val2017 이미지 경로: 139 → .../000000000139.jpg"""
    return COCO_IMAGE_DIR / f"{image_id:012d}.jpg"
