"""
케이스와 pool 읽기, 추론 항목 만들기
"""
import json

from mivlm.config import CASES_JSONL, POOLS_JSON


def load_cases() -> list[dict]:
    """data/cases/cases.jsonl의 케이스 전체 (한 줄에 케이스 하나)"""
    with open(CASES_JSONL) as f:
        return [json.loads(line) for line in f]


def load_pools() -> list[dict]:
    """data/cases/pools.json: 클래스마다 positives 8장과 negatives 8장"""
    with open(POOLS_JSON) as f:
        return json.load(f)


def single_image_items(pools: list[dict]) -> list[dict]:
    """이미지 1장 추론 항목: 클래스마다 P 8장, N 8장 순서로 만든다 (클래스 20개면 320개)

    label은 "이 이미지에 이 객체가 있는가"의 정답이다. P 이미지면 Yes, N 이미지면 No.
    같은 이미지가 두 클래스에 쓰이면 질문이 다르므로 서로 다른 항목이 된다.
    """
    items = []
    for pool in pools:
        for label, image_ids in [("Yes", pool["positives"]), ("No", pool["negatives"])]:
            for image_id in image_ids:
                items.append({
                    "item_id": f"{pool['name'].replace(' ', '_')}_{image_id}",
                    "category_id": pool["category_id"],
                    "object": pool["name"],
                    "image_id": image_id,
                    "label": label,
                })
    return items


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


def per_image_queries(cases: list[dict]) -> list[dict]:
    """이미지 3장을 넣고 한 자리를 지목해 묻는 항목: 케이스마다 자리 1, 2, 3에 대해 하나씩

    label은 "지목한 자리의 이미지에 객체가 있는가"의 정답이다. 그 자리가 P면 Yes, N이면 No.
    """
    queries = []
    for case in cases:
        for slot, letter in enumerate(case["variant"], start=1):
            queries.append({
                "query_id": f"{case['case_id']}_{slot}",
                "case_id": case["case_id"],
                "slot": slot,
                "image_ids": case["image_ids"],
                "object": case["object"],
                "variant": case["variant"],
                "label": "Yes" if letter == "P" else "No",
            })
    return queries
