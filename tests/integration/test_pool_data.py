"""
실제 케이스 파일로 하는 풀 읽기 검사 (기존 탐색 데이터와 dev만 읽는다. confirm과 eval은 읽지 않고, 모델도 쓰지 않는다)
pytest tests/integration/test_pool_data.py -v

1. 인자 없는 load_cases()와 data_sha256()이 예전과 같은 파일을 읽는다 (해시가 기존 실행의 meta.json과 같다)
2. dev 풀이 기존 데이터와 같은 구조로 읽히고, 해시가 생성 기록(new_pools_summary.json)과 같다
3. 저장된 기존 실행들(파일럿 제외)이 새 출처 확인(run_pool)을 통과한다. meta에 풀이 없으니 기존 데이터로 읽힌다
파일이 없으면 건너뛴다.
"""
import json
from collections import Counter

import pytest

from mivlm.config import CASES_DIR, CASES_JSONL, PREDICTIONS_DIR
from mivlm.data.cases import (
    data_sha256, items_for_cases, load_cases, load_pools, pilot_cases, run_pool, single_image_items,
)

ORIGINAL_SHA256 = {  # 기존 실행들의 meta.json에 남은 값
    "cases_sha256": "2c21311b716413ad8eaba6d361e086ea1c52864b95d4b084dca0ceb3de5a0af4",
    "pools_sha256": "7267e12eb8265234dc3e61999a8e6c190148c6e41a3d650fea32ed89695eb75f",
}
SUMMARY_JSON = CASES_DIR / "new_pools_summary.json"
EXISTING_RUNS = [  # 지금까지의 분석이 읽은 전체 실행
    "01_baseline/qwen2.5-vl-3b", "01_baseline/qwen2.5-vl-3b_float32", "01_baseline/qwen2.5-vl-7b",
    "02_decomposition/b_single_image/qwen2.5-vl-3b", "02_decomposition/b_single_image/qwen2.5-vl-7b",
    "02_decomposition/c_per_image_answers/qwen2.5-vl-3b", "02_decomposition/c_per_image_answers/qwen2.5-vl-7b",
    "03_hidden_states/qwen2.5-vl-3b", "03_hidden_states/qwen2.5-vl-3b_float32", "03_hidden_states/qwen2.5-vl-7b",
]

pytestmark = pytest.mark.skipif(not CASES_JSONL.exists(), reason="기존 data/cases 파일이 없음")


def test_the_default_still_reads_the_original_exploration_data():
    cases = load_cases()

    assert len(cases) == 2400
    assert data_sha256() == ORIGINAL_SHA256
    assert cases == [json.loads(line) for line in CASES_JSONL.read_text().splitlines()]


@pytest.mark.skipif(not (CASES_DIR / "dev" / "cases.jsonl").exists() or not SUMMARY_JSON.exists(),
                    reason="dev 풀이나 생성 기록이 없음")
def test_the_dev_pool_has_the_same_structure_and_the_recorded_hashes():
    base, dev = load_cases(), load_cases("dev")
    recorded = json.loads(SUMMARY_JSON.read_text())["pools"]["dev"]["files_sha256"]

    assert len(dev) == len(base) == 2400
    assert [list(case) for case in dev] == [list(case) for case in base]  # 필드와 그 순서가 같다
    assert [case["variant"] for case in dev] == [case["variant"] for case in base]  # 변형 순서가 같다
    assert data_sha256("dev")["cases_sha256"] == recorded["cases.jsonl"]
    assert data_sha256("dev")["pools_sha256"] == recorded["pools.json"]
    dev_images = {image_id for pool in load_pools("dev") for image_id in pool["positives"] + pool["negatives"]}
    assert {image_id for case in dev for image_id in case["image_ids"]} <= dev_images


@pytest.mark.parametrize("run", EXISTING_RUNS)
def test_existing_runs_pass_the_provenance_check(run):
    run_dir = PREDICTIONS_DIR / run
    if not (run_dir / "meta.json").exists():
        pytest.skip(f"{run} 실행 기록이 없음")

    assert run_pool(run_dir) is None


@pytest.mark.skipif(not (CASES_DIR / "dev" / "cases.jsonl").exists(), reason="dev 풀이 없음")
def test_the_dev_pilot_covers_every_class_and_the_four_variants_and_b_uses_the_same_images():
    pilot = pilot_cases(load_cases("dev"), "dev")
    items = items_for_cases(single_image_items(load_pools("dev")), pilot)

    assert len(pilot) == 20 and len({case["object"] for case in pilot}) == 20
    assert Counter(case["variant"] for case in pilot) == {"PPP": 5, "PPN": 5, "PNP": 5, "NPP": 5}
    assert len(items) == 60  # 케이스마다 이미지 3장
    pilot_images = {(case["object"], image_id) for case in pilot for image_id in case["image_ids"]}
    assert {(item["object"], item["image_id"]) for item in items} == pilot_images
