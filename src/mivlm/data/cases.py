"""
cases.jsonl 읽기
"""
import json

from mivlm.config import CASES_JSONL


def load_cases() -> list[dict]:
    """data/cases/cases.jsonl의 케이스 전체 (한 줄에 케이스 하나)"""
    with open(CASES_JSONL) as f:
        return [json.loads(line) for line in f]


def pilot_cases(cases: list[dict]) -> list[dict]:
    """파일럿용 케이스: 클래스마다 첫 pair(group_01의 pair 1)에서 하나씩 고른다

    변형은 클래스 순서대로 돌아가며 고른다 (1번째 클래스는 PPP, 2번째는 PPN, …).
    그래서 모든 클래스와 모든 변형이 섞여 들어간다.
    """
    first_pairs = {}  # object -> 그 클래스 첫 pair의 케이스들 (파일에 적힌 변형 순서 그대로)
    for case in cases:
        if case["group_id"].startswith("group_01_") and case["pair_id"] == 1:
            first_pairs.setdefault(case["object"], []).append(case)
    return [variants[i % len(variants)] for i, variants in enumerate(first_pairs.values())]
