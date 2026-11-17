"""scripts/download_coco.py 단위 테스트"""
import json
import pytest

from download_coco import count_images, download, summarize_annotations


# ---------------------------------------------------------------- count_images

def test_count_images_counts_only_jpg(tmp_path):
    for name in ["a.jpg", "b.jpg", "c.jpg", "note.txt"]:
        (tmp_path / name).touch()

    assert count_images(tmp_path) == 3

def test_count_images_missing_dir(tmp_path):
    assert count_images(tmp_path / "nope") == 0

# ------------------------------------------------------- summarize_annotations

def test_summarize_annotations(tmp_path):
    # 이미지 2장 중 annotation은 전부 이미지 1에 몰려 있다 → n_images_with_ann == 1
    # category id가 1, 3으로 연속이 아닌 것도 실제 COCO와 같은 상황이다.
    data = {
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1},
            {"id": 11, "image_id": 1, "category_id": 1},
            {"id": 12, "image_id": 1, "category_id": 3},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
    }
    json_path = tmp_path / "instances.json"
    json_path.write_text(json.dumps(data))

    assert summarize_annotations(json_path) == {
        "n_images": 2,
        "n_annotations": 3,
        "n_categories": 2,
        "n_images_with_ann": 1,
    }

# -------------------------------------------------------------------- download

def test_download_creates_file(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 1000)
    dest = tmp_path / "dest.bin"

    download(src.as_uri(), dest, expected_size=1000)

    assert dest.read_bytes() == src.read_bytes()
    assert not (tmp_path / "dest.bin.part").exists()

def test_download_skips_when_complete(tmp_path):
    dest = tmp_path / "dest.bin"
    dest.write_bytes(b"x" * 1000)
    missing_url = (tmp_path / "does_not_exist.bin").as_uri()

    download(missing_url, dest, expected_size=1000)

    assert dest.stat().st_size == 1000

def test_download_size_mismatch_raises(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 1000)
    dest = tmp_path / "dest.bin"

    with pytest.raises(ValueError):
        download(src.as_uri(), dest, expected_size=999)

    assert not dest.exists()
    assert not (tmp_path / "dest.bin.part").exists()
