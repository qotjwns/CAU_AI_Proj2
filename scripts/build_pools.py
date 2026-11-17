"""
COCO-ReM 주석으로 클래스 20개를 고르고, 클래스마다 P 8장 / N 8장을 뽑아 pools.json으로 저장
python scripts/build_pools.py
"""
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INSTANCES_JSON = PROJECT_ROOT / "data" / "coco" / "annotations" / "instances_valrem.json"
POOLS_JSON = PROJECT_ROOT / "data" / "cases" / "pools.json"
MIN_AREA_RATIO = 0.05
N_CLASSES = 20
POOL_SIZE = 8
SEED = 42


def find_candidates(coco: dict) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """클래스별 P 후보, N 후보 image id 목록 (image id 오름차순)

    P: crowd가 아닌 해당 클래스 bbox 중 하나 이상이 이미지 면적의 MIN_AREA_RATIO 이상
    N: 해당 클래스 주석이 크기와 상관없이 하나도 없음
    """
    image_area = {img["id"]: img["width"] * img["height"] for img in coco["images"]}
    category_ids = [cat["id"] for cat in coco["categories"]]
    present = {cid: set() for cid in category_ids}
    large = {cid: set() for cid in category_ids}

    for ann in coco["annotations"]:
        cid, image_id = ann["category_id"], ann["image_id"]
        present[cid].add(image_id)
        _, _, w, h = ann["bbox"]
        if not ann["iscrowd"] and w * h >= MIN_AREA_RATIO * image_area[image_id]:
            large[cid].add(image_id)

    all_images = set(image_area)
    positives = {cid: sorted(large[cid]) for cid in category_ids}
    negatives = {cid: sorted(all_images - present[cid]) for cid in category_ids}
    return positives, negatives


def select_classes(positives: dict[int, list[int]], negatives: dict[int, list[int]]) -> list[int]:
    """P 후보와 N 후보가 모두 POOL_SIZE 이상인 클래스를 P 후보가 많은 순(같으면 category id 순)으로 N_CLASSES개"""
    eligible = [
        cid for cid in positives
        if len(positives[cid]) >= POOL_SIZE and len(negatives[cid]) >= POOL_SIZE
    ]
    eligible.sort(key=lambda cid: (-len(positives[cid]), cid))
    return eligible[:N_CLASSES]


def main() -> None:
    with open(INSTANCES_JSON) as f:
        coco = json.load(f)
    names = {cat["id"]: cat["name"] for cat in coco["categories"]}

    positives, negatives = find_candidates(coco)
    pools = []
    for rank, cid in enumerate(select_classes(positives, negatives), start=1):
        pools.append({
            "category_id": cid,
            "name": names[cid],
            "n_positive_candidates": len(positives[cid]),
            "n_negative_candidates": len(negatives[cid]),
            "positives": random.Random(f"{SEED}:{cid}:positive_candidates").sample(positives[cid], POOL_SIZE),
            "negatives": random.Random(f"{SEED}:{cid}:negative_candidates").sample(negatives[cid], POOL_SIZE),
        })
        print(f"{rank:2d}  {names[cid]:<14} P {len(positives[cid]):5d}  N {len(negatives[cid]):5d}")

    POOLS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(POOLS_JSON, "w") as f:
        json.dump(pools, f, indent=2)
    print(f"[ok] {len(pools)} classes -> {POOLS_JSON}")


if __name__ == "__main__":
    main()
