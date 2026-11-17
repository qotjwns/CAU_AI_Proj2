"""
cases.jsonl 읽기
"""
import json

from mivlm.config import CASES_JSONL


def load_cases() -> list[dict]:
    """data/cases/cases.jsonl의 케이스 전체 (한 줄에 케이스 하나)"""
    with open(CASES_JSONL) as f:
        return [json.loads(line) for line in f]
