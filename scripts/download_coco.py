"""
COCO val2017 이미지와 COCO-ReM annotation 다운로드
python scripts/download_coco.py
"""
import json
import urllib.request
import zipfile
from pathlib import Path

from mivlm.config import COCO_IMAGE_DIR, COCO_ROOT, INSTANCES_JSON

IMAGES_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
IMAGES_ZIP_BYTES = 815_585_330
ANNOTATIONS_ZIP_URL = "https://huggingface.co/datasets/kdexd/coco-rem/resolve/main/instances_valrem.json.zip"
ANNOTATIONS_ZIP_BYTES = 10_410_664
N_VAL_IMAGES = 5000
CHUNK_SIZE = 1 << 20  # 1 MB


def count_images(image_dir: Path) -> int:
    """image_dir 바로 아래의 .jpg 파일 수 count"""
    if not image_dir.exists():
        return 0
    return sum(1 for _ in image_dir.glob("*.jpg"))

def summarize_annotations(json_path: Path) -> dict:
    """COCO instances json의 이미지 / annotation / 카테고리 개수를 요약"""
    with open(json_path) as f:
        data = json.load(f)
    return {
        "n_images": len(data["images"]),
        "n_annotations": len(data["annotations"]),
        "n_categories": len(data["categories"]),
        "n_images_with_ann": len({a["image_id"] for a in data["annotations"]}),
    }


def download(url: str, dest: Path, expected_size: int) -> None:
    """url을 dest로 다운로드"""
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"[skip] {dest.name} (already downloaded)")
        return

    part = dest.with_name(dest.name + ".part")
    done = 0
    with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as f:
        while chunk := resp.read(CHUNK_SIZE):
            f.write(chunk)
            done += len(chunk)
            print(f"\r[download] {dest.name}: {done / expected_size:.1%}", end="", flush=True)
    print()

    actual = part.stat().st_size
    if actual != expected_size:
        part.unlink()
        raise ValueError(f"{dest.name}: expected {expected_size} bytes, got {actual}")
    part.replace(dest)


def main() -> None:
    zips_dir = COCO_ROOT / "zips"
    images_zip = zips_dir / "val2017.zip"
    annotations_zip = zips_dir / "instances_valrem.json.zip"

    if count_images(COCO_IMAGE_DIR) == N_VAL_IMAGES and INSTANCES_JSON.exists():
        print(f"[skip] COCO val2017 already prepared at {COCO_ROOT}")
    else:
        zips_dir.mkdir(parents=True, exist_ok=True)
        download(IMAGES_ZIP_URL, images_zip, IMAGES_ZIP_BYTES)
        download(ANNOTATIONS_ZIP_URL, annotations_zip, ANNOTATIONS_ZIP_BYTES)

        print("[extract] val2017.zip")
        with zipfile.ZipFile(images_zip) as zf:
            zf.extractall(COCO_ROOT)
        print(f"[extract] {INSTANCES_JSON.name}")

        with zipfile.ZipFile(annotations_zip) as zf:
            zf.extractall(INSTANCES_JSON.parent, members=[INSTANCES_JSON.name])

    n_images = count_images(COCO_IMAGE_DIR)
    if n_images != N_VAL_IMAGES:
        raise RuntimeError(f"expected {N_VAL_IMAGES} images in {COCO_IMAGE_DIR}, found {n_images}")

    print(f"[ok] images on disk: {n_images}")
    for key, value in summarize_annotations(INSTANCES_JSON).items():
        print(f"[ok] {key}: {value}")


if __name__ == "__main__":
    main()
