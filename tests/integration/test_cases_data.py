"""
실제 생성 파일(pools.json → pairs.json → cases.jsonl)이 서로 맞게 이어졌는지 검사
pytest tests/integration/test_cases_data.py -v

데이터를 다시 만들 때마다 실행한다. 파일이 없으면 건너뛴다.
"""
import json
from collections import Counter

import pytest

from mivlm.config import CASES_JSONL, PAIRS_JSON, POOLS_JSON
from mivlm.data.cases import load_cases

pytestmark = pytest.mark.skipif(
    not (POOLS_JSON.exists() and PAIRS_JSON.exists() and CASES_JSONL.exists()),
    reason="data/cases 파일이 없음: build_pools.py → build_pairs.py → build_cases.py를 먼저 실행",
)

VARIANT_NAMES = ["PPP", "PPN", "PNN", "NNN", "NPP", "PNP"]


@pytest.fixture(scope="module")
def pools():
    with open(POOLS_JSON) as f:
        return {pool["category_id"]: pool for pool in json.load(f)}


@pytest.fixture(scope="module")
def pairs():
    with open(PAIRS_JSON) as f:
        return {(pair["group_id"], pair["pair_id"]): pair for pair in json.load(f)}


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def test_variant_letters_match_pool_membership(pools, cases):
    # 자리마다: 글자가 P면 그 클래스 pool의 P 이미지, N이면 N 이미지여야 한다
    for case in cases:
        pool = pools[case["category_id"]]
        letters = ""
        for image_id in case["image_ids"]:
            if image_id in pool["positives"]:
                letters += "P"
            elif image_id in pool["negatives"]:
                letters += "N"
            else:
                letters += "?"
        assert letters == case["variant"], case["case_id"]


def test_cases_come_from_current_pairs_file(pairs, cases):
    # pairs.json만 다시 만들고 cases.jsonl을 안 만들면 여기서 걸린다
    assert len(cases) == len(pairs) * len(VARIANT_NAMES)
    for case in cases:
        pair = pairs[(case["group_id"], case["pair_id"])]
        assert set(case["image_ids"]) <= set(pair["p"] + pair["n"]), case["case_id"]


def test_each_variant_has_400(cases):
    assert Counter(case["variant"] for case in cases) == {name: 400 for name in VARIANT_NAMES}


def test_mode_counts(cases):
    assert sum("A" in case["modes"] for case in cases) == 1600
    assert sum("B" in case["modes"] for case in cases) == 1200


def test_only_ppp_is_yes(cases):
    for case in cases:
        assert case["label"] == ("Yes" if case["variant"] == "PPP" else "No"), case["case_id"]
    assert Counter(case["label"] for case in cases) == {"Yes": 400, "No": 2000}


def test_case_ids_are_unique(cases):
    assert len({case["case_id"] for case in cases}) == len(cases)


def test_mode_b_moves_the_same_negative(cases):
    # 같은 pair의 NPP, PNP, PPN에서 N 자리에 있는 이미지는 모두 같은 n1이어야 한다
    by_pair = {}
    for case in cases:
        by_pair.setdefault((case["group_id"], case["pair_id"]), {})[case["variant"]] = case["image_ids"]
    for key, variants in by_pair.items():
        assert variants["NPP"][0] == variants["PNP"][1] == variants["PPN"][2], key
