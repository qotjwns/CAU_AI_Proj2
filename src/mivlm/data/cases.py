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


def case_pairs(cases: list[dict]) -> list[dict]:
    """케이스를 pair별로 묶는다 (pair도, pair 안의 변형도 처음 나온 순서 유지)

    반환: [{"pair": "group_01_person_1", "cases": {"PPP": 케이스, "PPN": 케이스, ...}}, ...]
    pair 이름은 group_id와 pair_id를 이은 것이고, case_id에서 변형 이름을 뗀 것과 같다.
    """
    pairs = {}  # pair 이름 -> {변형: 케이스}
    for case in cases:
        pairs.setdefault(f"{case['group_id']}_{case['pair_id']}", {})[case["variant"]] = case
    return [{"pair": pair, "cases": variants} for pair, variants in pairs.items()]


def image_linked_classes(cases: list[dict]) -> list[set[str]]:
    """이미지를 같이 쓰는 클래스끼리 한 덩어리로 묶는다: 덩어리(클래스 이름의 set)의 목록, 처음 나온 순서

    a와 b가 이미지 하나를 같이 쓰고 b와 c가 다른 이미지를 같이 쓰면 a, b, c가 한 덩어리다.
    어떤 클래스와도 이미지를 같이 쓰지 않는 클래스는 혼자 덩어리가 된다. 서로 다른 덩어리는 이미지를 하나도 공유하지 않는다.
    """
    group_of = {}  # 클래스 -> 그 클래스가 속한 덩어리. 같은 덩어리의 클래스들은 같은 set을 가리킨다
    for case in cases:
        group_of.setdefault(case["object"], {case["object"]})

    classes_of_image = {}  # 이미지 -> 그 이미지를 쓰는 클래스들
    for case in cases:
        for image_id in case["image_ids"]:
            classes_of_image.setdefault(image_id, set()).add(case["object"])
    for classes in classes_of_image.values():
        merged = set().union(*(group_of[name] for name in classes))
        for name in merged:
            group_of[name] = merged

    groups = []  # 덩어리마다 한 번씩, 처음 나온 순서
    for group in group_of.values():
        if not any(group is seen for seen in groups):
            groups.append(group)
    return groups
