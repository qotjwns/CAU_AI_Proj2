"""
기존 데이터 파일(pools.json, pairs.json, cases.jsonl)이 scripts/build_new_pools.py의 함수로 바이트까지 그대로 다시 만들어지는지 검사
pytest tests/integration/test_build_new_pools_data.py -v

새 풀은 만들지 않는다. 기존 절차(seed 접두어 "42", 제외 없음)만 메모리에서 다시 계산해 저장된 파일과 비교한다.
실제 COCO-ReM 주석과 기존 data/cases 파일이 있어야 하고, 없으면 건너뛴다.
"""
import json

import pytest

from build_new_pools import pool_files, pool_images, sample_pool, sha256_text
from build_pools import find_candidates, select_classes
from mivlm.config import CASES_JSONL, INSTANCES_JSON, PAIRS_JSON, POOLS_JSON

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in [INSTANCES_JSON, POOLS_JSON, PAIRS_JSON, CASES_JSONL]),
    reason="COCO-ReM 주석이나 기존 data/cases 파일이 없음",
)
ORIGINAL_CASES_SHA256 = "2c21311b716413ad8eaba6d361e086ea1c52864b95d4b084dca0ceb3de5a0af4"  # 기존 실행들의 meta.json


def test_the_original_files_are_rebuilt_byte_for_byte():
    with open(INSTANCES_JSON) as f:
        coco = json.load(f)
    positives, negatives = find_candidates(coco)
    names = {category["id"]: category["name"] for category in coco["categories"]}
    classes = [(cid, names[cid]) for cid in select_classes(positives, negatives)]

    files = pool_files(sample_pool("42", classes, positives, negatives, set()), "42")

    assert files["pools.json"] == POOLS_JSON.read_text()
    assert files["pairs.json"] == PAIRS_JSON.read_text()
    assert files["cases.jsonl"] == CASES_JSONL.read_text()
    assert sha256_text(files["cases.jsonl"]) == ORIGINAL_CASES_SHA256


def test_the_base_pools_hold_the_311_images_to_exclude():
    base_pools = json.loads(POOLS_JSON.read_text())

    assert len(base_pools) == 20
    assert len(pool_images(base_pools)) == 311
