"""
pools.json의 클래스별 P 8장 / N 8장으로 group 100개, pair 400개를 만들어 pairs.json으로 저장
python scripts/build_pairs.py
"""
import itertools
import json
import random

from build_pools import POOLS_JSON, PROJECT_ROOT, SEED

PAIRS_JSON = PROJECT_ROOT / "data" / "cases" / "pairs.json"
GROUP_PATTERNS = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 2, 4, 6), (1, 3, 5, 7), (0, 3, 4, 7)]
PAIR_SIZE = 3


def make_pairs(pool: dict) -> list[dict]:
    """클래스 pool 하나로 group 5개 × pair 4개 = pair 20개를 만든다"""
    pairs = []
    for g, pattern in enumerate(GROUP_PATTERNS, start=1):
        group_id = f"group_{g:02d}_{pool['name'].replace(' ', '_')}"
        group_positives = [pool["positives"][i] for i in pattern]
        rng = random.Random(f"{SEED}:{group_id}:pairs")
        # combinations는 입력 순서를 유지한다: [0,1,2] → [0,1,3] → [0,2,3] → [1,2,3]
        for pair_id, p in enumerate(itertools.combinations(group_positives, PAIR_SIZE), start=1):
            pairs.append({
                "group_id": group_id,
                "pair_id": pair_id,
                "category_id": pool["category_id"],
                "object": pool["name"],
                "p": list(p),
                "n": rng.sample(pool["negatives"], PAIR_SIZE),
            })
    return pairs


def main() -> None:
    with open(POOLS_JSON) as f:
        pools = json.load(f)

    pairs = []
    for pool in pools:
        class_pairs = make_pairs(pool)
        pairs.extend(class_pairs)
        print(f"{pool['name']:<14} {len(class_pairs)} pairs")

    with open(PAIRS_JSON, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"[ok] {len(pairs)} pairs -> {PAIRS_JSON}")


if __name__ == "__main__":
    main()
