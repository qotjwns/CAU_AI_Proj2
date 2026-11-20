"""
케이스와 pool 읽기, 추론 항목 만들기

데이터 출처: 풀 이름(None이면 기존 탐색 데이터, 그 밖에는 mivlm.config.POOL_NAMES)이 케이스 파일의 폴더를 정한다 (pool_dir).
실행은 meta.json에 풀 이름(풀을 쓴 경우만)과 케이스·풀 파일의 SHA-256을 남긴다 (mivlm.runner).
분석은 실행 폴더의 meta.json에서 풀을 읽고, 저장된 해시가 지금 파일과 같은지 확인한 뒤 케이스를 읽는다 (run_pool).
"""
import hashlib
import json
from pathlib import Path

from mivlm.config import POOL_NAMES, pool_dir
from mivlm.run_files import read_meta

HASH_KEYS = {"cases_sha256": "cases.jsonl", "pools_sha256": "pools.json"}  # meta.json의 해시 키 -> 그 파일 이름


def _data_name(pool: str | None) -> str:
    return "기존 탐색 데이터" if pool is None else f"풀 {pool}"


def pool_file(pool: str | None, name: str) -> Path:
    """풀의 케이스 파일 경로. 파일이 없으면 무엇으로 만드는지를 담아 FileNotFoundError를 낸다"""
    path = pool_dir(pool) / name
    if not path.exists():
        builder = ("scripts/build_pools.py → build_pairs.py → build_cases.py" if pool is None
                   else "scripts/build_new_pools.py")
        raise FileNotFoundError(f"{_data_name(pool)}의 {name}이 없음: {path}. {builder}로 만든다")
    return path


def load_cases(pool: str | None = None) -> list[dict]:
    """케이스 전체 (cases.jsonl, 한 줄에 케이스 하나, 파일 순서 그대로). pool이 None이면 기존 탐색 데이터 data/cases/"""
    with open(pool_file(pool, "cases.jsonl")) as f:
        return [json.loads(line) for line in f]


def load_pools(pool: str | None = None) -> list[dict]:
    """pools.json: 클래스마다 positives 8장과 negatives 8장. pool이 None이면 기존 탐색 데이터"""
    with open(pool_file(pool, "pools.json")) as f:
        return json.load(f)


def data_sha256(pool: str | None = None) -> dict[str, str]:
    """풀의 케이스 파일과 풀 파일의 SHA-256: {"cases_sha256", "pools_sha256"}. 실행의 meta.json에 같은 키로 남긴다"""
    return {key: hashlib.sha256(pool_file(pool, name).read_bytes()).hexdigest() for key, name in HASH_KEYS.items()}


def run_pool(run_dir: Path) -> str | None:
    """실행 폴더의 meta.json이 가리키는 풀을 검증해서 돌려준다

    "pool" 키가 없으면 기존 탐색 데이터(None)다 (풀이 생기기 전의 실행). 키가 있으면 등록된 풀 이름이어야 하고,
    아니면 ValueError를 낸다. 기존 데이터로 바꿔 읽지 않는다.
    meta.json에 cases_sha256, pools_sha256이 있으면 지금 파일의 해시와 같아야 한다. 다르면 ValueError.
    실행 당시의 데이터와 분석하는 지금의 데이터가 같다는 것을 이렇게 확인한다. 해시가 없는 예전 실행은 확인하지 않는다.
    """
    meta = read_meta(run_dir)
    pool = meta.get("pool")
    if "pool" in meta and pool not in POOL_NAMES:
        raise ValueError(f"{run_dir}의 meta.json에 알 수 없는 풀 이름: {pool!r}. 쓸 수 있는 이름은 {POOL_NAMES}")
    current = data_sha256(pool)
    changed = [key for key in HASH_KEYS if key in meta and meta[key] != current[key]]
    if changed:
        raise ValueError(f"{run_dir}의 실행 당시 데이터와 지금 {_data_name(pool)}의 파일이 다름: {changed}")
    return pool


def load_run_cases(run_dir: Path) -> list[dict]:
    """실행 폴더가 쓴 풀의 케이스 (run_pool로 풀과 해시를 확인한 뒤 읽는다)"""
    return load_cases(run_pool(run_dir))


def shared_run_pool(run_dirs: list[Path]) -> str | None:
    """여러 실행(예: A, B, C)이 같은 데이터에서 나왔는지 확인하고 그 풀을 돌려준다

    실행마다 run_pool로 검증한 뒤, 풀 이름이 모두 같고 meta.json에 남은 해시(있는 것끼리)가 모두 같아야 한다.
    다르면 ValueError. 결과를 케이스에 붙이는 일은 id로 하므로(join_results) 행 순서에 기대지 않는다.
    """
    pools = [run_pool(run_dir) for run_dir in run_dirs]
    if len(set(pools)) != 1:
        raise ValueError(f"실행마다 풀이 다름: {dict(zip(map(str, run_dirs), pools))}")
    metas = [read_meta(run_dir) for run_dir in run_dirs]
    for key in HASH_KEYS:
        recorded = {meta[key] for meta in metas if key in meta}
        if len(recorded) > 1:
            raise ValueError(f"실행마다 {key}가 다름: {[str(run_dir) for run_dir in run_dirs]}")
    return pools[0]


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


POOL_PILOT_VARIANTS = ["PPP", "PPN", "PNP", "NPP"]  # 새 풀의 파일럿에서 돌아가며 고르는 변형 (N이 없거나 하나)


def pilot_cases(cases: list[dict], pool: str | None = None) -> list[dict]:
    """파일럿용 케이스: 클래스마다 첫 pair(group_01의 pair 1)에서 하나씩 고른다

    변형은 클래스 순서대로 돌아가며 고른다 (1번째 클래스는 첫 변형, 2번째는 둘째 변형, …). 무작위가 없다.
    기존 탐색 데이터(pool None)는 그 pair의 변형 6개를 파일 순서대로 돌아가며 고른다 (예전과 같다).
    새 풀은 POOL_PILOT_VARIANTS에 든 변형만 파일 순서대로 돌아가며 고른다. 클래스가 20개면 변형마다 5개다.
    """
    first_pairs = {}  # object -> 그 클래스 첫 pair의 케이스들 (파일에 적힌 변형 순서 그대로)
    for case in cases:
        if (case["group_id"].startswith("group_01_") and case["pair_id"] == 1
                and (pool is None or case["variant"] in POOL_PILOT_VARIANTS)):
            first_pairs.setdefault(case["object"], []).append(case)
    return [variants[i % len(variants)] for i, variants in enumerate(first_pairs.values())]


def items_for_cases(items: list[dict], cases: list[dict]) -> list[dict]:
    """이미지 1장 항목(single_image_items) 중 cases에 쓰인 (클래스, 이미지)에 해당하는 것 (items 순서 그대로)

    파일럿에서 B(이미지 1장씩)를 A의 파일럿 케이스와 같은 이미지로 돌리는 데 쓴다.
    """
    wanted = {(case["object"], image_id) for case in cases for image_id in case["image_ids"]}
    return [item for item in items if (item["object"], item["image_id"]) in wanted]


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
