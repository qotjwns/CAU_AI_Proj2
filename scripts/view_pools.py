"""
pools.json의 클래스별 P 8장 / N 8장을 클래스마다 그림 한 장으로 저장
python scripts/view_pools.py
"""
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from build_pools import INSTANCES_JSON, POOLS_JSON, PROJECT_ROOT

IMAGE_DIR = PROJECT_ROOT / "data" / "coco" / "val2017"
OUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "pools"


def main() -> None:
    with open(POOLS_JSON) as f:
        pools = json.load(f)
    with open(INSTANCES_JSON) as f:
        coco = json.load(f)

    image_area = {img["id"]: img["width"] * img["height"] for img in coco["images"]}
    boxes = {}  # (image_id, category_id) -> [(bbox, iscrowd), ...]
    for ann in coco["annotations"]:
        boxes.setdefault((ann["image_id"], ann["category_id"]), []).append((ann["bbox"], ann["iscrowd"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rank, pool in enumerate(pools, start=1):
        fig, axes = plt.subplots(2, 8, figsize=(24, 6.5))
        for row, (label, key) in enumerate([("P", "positives"), ("N", "negatives")]):
            for col, image_id in enumerate(pool[key]):
                ax = axes[row, col]
                # 흑백 이미지는 2차원 배열이라 cmap을 지정하지 않으면 컬러맵이 입혀진다
                ax.imshow(plt.imread(IMAGE_DIR / f"{image_id:012d}.jpg"), cmap="gray")
                # 해당 클래스 bbox 전부: 빨강 = 일반(면적 %), 노랑 = crowd (P 판정에서 제외됨)
                for (x, y, w, h), iscrowd in boxes.get((image_id, pool["category_id"]), []):
                    color = "yellow" if iscrowd else "red"
                    text = "crowd" if iscrowd else f"{w * h / image_area[image_id]:.1%}"
                    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2))
                    ax.text(x, y, text, fontsize=8, color="black", backgroundcolor=color)
                ax.set_title(f"{label} {image_id}", fontsize=9)
                ax.axis("off")

        fig.suptitle(
            f"{rank:02d} {pool['name']}  "
            f"(P candidates {pool['n_positive_candidates']}, N candidates {pool['n_negative_candidates']})"
        )
        out_path = OUT_DIR / f"{rank:02d}_{pool['name'].replace(' ', '_')}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
