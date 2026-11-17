"""
COCO val2017 이미지 경로
이미지 경로 생성
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]  # image.py → data → mivlm → src → 프로젝트 루트
COCO_IMAGE_DIR = PROJECT_ROOT / "data" / "coco" / "val2017"


def image_path(image_id: int) -> Path:
    """COCO val2017 이미지 경로: 139 → .../000000000139.jpg"""
    return COCO_IMAGE_DIR / f"{image_id:012d}.jpg"
