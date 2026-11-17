"""coco val 2017 이미지, annotation 다운로드"""
import json
import urllib.request
import zipfile
from pathlib import Path

IMAGES_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
IMAGES_ZIP_BYTES = 815_585_330
ANNOTATIONS_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANNOTATIONS_ZIP_BYTES = 252_907_541
INSTANCES_JSON = "annotations/instances_val2017.json"
N_VAL_IMAGES = 5000
CHUNK_SIZE = 1 << 20  # 1 MB
COCO_ROOT = Path(__file__).resolve().parents[1] / "data" / "coco"


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
    image_dir = COCO_ROOT / "val2017"
    instances_json_path = COCO_ROOT / INSTANCES_JSON
    zips_dir = COCO_ROOT / "zips"
    images_zip = zips_dir / "val2017.zip"
    annotations_zip = zips_dir / "annotations_trainval2017.zip"

    if count_images(image_dir) == N_VAL_IMAGES and instances_json_path.exists():
        print(f"[skip] COCO val2017 already prepared at {COCO_ROOT}")
    else:
        zips_dir.mkdir(parents=True, exist_ok=True)
        download(IMAGES_ZIP_URL, images_zip, IMAGES_ZIP_BYTES)
        download(ANNOTATIONS_ZIP_URL, annotations_zip, ANNOTATIONS_ZIP_BYTES)

        print("[extract] val2017.zip")
        with zipfile.ZipFile(images_zip) as zf:
            zf.extractall(COCO_ROOT)
        print(f"[extract] {INSTANCES_JSON}")
        with zipfile.ZipFile(annotations_zip) as zf:
            zf.extractall(COCO_ROOT, members=[INSTANCES_JSON])

    n_images = count_images(image_dir)
    if n_images != N_VAL_IMAGES:
        raise RuntimeError(f"expected {N_VAL_IMAGES} images in {image_dir}, found {n_images}")

    print(f"[ok] images on disk: {n_images}")
    for key, value in summarize_annotations(instances_json_path).items():
        print(f"[ok] {key}: {value}")


if __name__ == "__main__":
    main()
