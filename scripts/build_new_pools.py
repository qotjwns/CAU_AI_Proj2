"""
독립 확증용 새 이미지 풀 세 개(dev, confirm, eval)를 사전 등록(docs/preregistration.md 2.2절, 커밋 1382ac5)대로 만든다
python scripts/build_new_pools.py

입력: COCO-ReM 주석(instances_valrem.json)과 기존 data/cases/pools.json (클래스 목록, 제외할 기존 이미지)
출력: data/cases/{dev,confirm,eval}/{pools.json, pairs.json, cases.jsonl}와 data/cases/new_pools_summary.json
    출력 중 하나라도 이미 있으면 아무것도 쓰지 않고 멈춘다. 만든 풀이 다른 코드로 다시 만들어지는 것을 막는다.
    커밋하지 않은 변경이 있어도 멈춘다. 풀을 만드는 코드는 풀을 만들기 전에 커밋해야 한다 (사전 등록 2.2절).

절차 (사전 등록 그대로)
    클래스: 기존 pools.json의 category id를 같은 순서로 쓴다. 다시 고르지 않는다
    후보: build_pools.find_candidates와 같은 정의 (P: crowd가 아닌 그 클래스 bbox가 이미지 면적의 5% 이상, N: 그 클래스 주석이 없음)
    제외: 기존 풀의 이미지 전부와, 먼저 만든 새 풀의 이미지 전부 (역할과 클래스에 상관없이). 순서는 dev → confirm → eval
    뽑기: 제외 뒤 후보(image id 오름차순)에서 random.Random(f"42:{풀}:{category id}:positive_candidates").sample(후보, 8),
        N은 f"42:{풀}:{category id}:negative_candidates"로 같은 방식. pair의 N 3장은 make_pairs(pool, f"42:{풀}")
    한 풀 안에서는 클래스마다 따로 뽑으므로 한 이미지가 여러 클래스에 쓰일 수 있다. 그 이미지는 요약에 남긴다
세 풀을 모두 메모리에서 만들고 점검(check_new_pools)을 통과한 뒤에만 파일을 쓴다 (write_new_pools).

봉인 평가 풀(eval)은 만들고 점검만 한다. 이 스크립트는 image id 목록을 출력하지 않는다.
"""
import hashlib
import itertools
import json
import platform
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

from build_cases import make_cases
from build_pairs import make_pairs
from build_pools import MIN_AREA_RATIO, POOL_SIZE, SEED, find_candidates
from mivlm.config import CASES_DIR, INSTANCES_JSON, POOLS_JSON
from mivlm.runner import git_state

POOL_NAMES = ["dev", "confirm", "eval"]  # 만드는 순서. 앞 풀의 이미지는 뒤 풀의 후보에서 빠진다
FILE_NAMES = ["pools.json", "pairs.json", "cases.jsonl"]
SUMMARY_NAME = "new_pools_summary.json"
TEMPORARY_NAME = ".new_pools_tmp"  # 파일을 다 쓴 뒤 제자리로 옮기기 전까지 쓰는 임시 폴더


def pool_images(pools: list[dict]) -> set[int]:
    """풀(클래스마다의 항목 목록)에 들어 있는 모든 image id. P와 N, 클래스를 구분하지 않는다"""
    return {image_id for pool in pools for image_id in pool["positives"] + pool["negatives"]}


def sample_pool(seed_prefix: str, classes: list[tuple[int, str]], positives: dict[int, list[int]],
                negatives: dict[int, list[int]], excluded: set[int]) -> list[dict]:
    """클래스마다 P와 N을 POOL_SIZE장씩 뽑은 풀: 기존 pools.json과 같은 모양의 목록 (classes 순서)

    classes: [(category id, 이름), ...]. positives, negatives: {category id: 후보 image id (오름차순)}
    excluded의 이미지는 후보에서 뺀다. 뺀 뒤의 후보 수를 n_positive_candidates, n_negative_candidates에 적는다.
    seed는 f"{seed_prefix}:{category id}:positive_candidates"와 f"{seed_prefix}:{category id}:negative_candidates"다.
    seed_prefix가 "42"이고 excluded가 비어 있으면 기존 build_pools.py와 같은 결과가 나온다.
    후보가 POOL_SIZE보다 적은 클래스가 하나라도 있으면, 그 클래스들과 모자란 수를 담아 ValueError를 낸다.
    """
    remaining = {cid: ([image_id for image_id in positives[cid] if image_id not in excluded],
                       [image_id for image_id in negatives[cid] if image_id not in excluded])
                 for cid, _ in classes}
    short = []  # "클래스: P 5장 (모자람 3)" 모양. 모자란 역할만 적는다
    for cid, name in classes:
        lacking = [f"{role} {len(candidates)}장 (모자람 {POOL_SIZE - len(candidates)})"
                   for role, candidates in zip(["P", "N"], remaining[cid]) if len(candidates) < POOL_SIZE]
        if lacking:
            short.append(f"{name}: " + ", ".join(lacking))
    if short:
        raise ValueError(f"후보가 {POOL_SIZE}장보다 적은 클래스가 있어 풀을 만들지 않는다 (seed {seed_prefix}):\n"
                         + "\n".join(short))
    pools = []
    for cid, name in classes:
        p_candidates, n_candidates = remaining[cid]
        pools.append({
            "category_id": cid,
            "name": name,
            "n_positive_candidates": len(p_candidates),
            "n_negative_candidates": len(n_candidates),
            "positives": random.Random(f"{seed_prefix}:{cid}:positive_candidates").sample(p_candidates, POOL_SIZE),
            "negatives": random.Random(f"{seed_prefix}:{cid}:negative_candidates").sample(n_candidates, POOL_SIZE),
        })
    return pools


def build_new_pools(coco: dict, base_pools: list[dict], names: list[str] = POOL_NAMES) -> dict[str, list[dict]]:
    """새 풀들을 names 순서로 만든다: {풀 이름: 풀}. 제외 집합은 기존 풀의 이미지에서 시작해 풀을 만들 때마다 커진다"""
    positives, negatives = find_candidates(coco)
    classes = [(pool["category_id"], pool["name"]) for pool in base_pools]
    excluded = pool_images(base_pools)
    built = {}
    for name in names:
        built[name] = sample_pool(f"{SEED}:{name}", classes, positives, negatives, excluded)
        excluded = excluded | pool_images(built[name])
    return built


def annotation_violations(coco: dict, pools: list[dict]) -> list[str]:
    """뽑힌 P와 N이 주석의 정의를 지키는지 find_candidates와 따로 다시 본다: 어긴 경우의 설명 목록 (없으면 빈 목록)

    P: 그 클래스의 crowd가 아닌 bbox 중 하나 이상이 이미지 면적의 MIN_AREA_RATIO 이상. N: 그 클래스 주석이 하나도 없음
    """
    image_area = {image["id"]: image["width"] * image["height"] for image in coco["images"]}
    annotations = {}  # image id -> 그 이미지의 주석들
    for annotation in coco["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)
    violations = []
    for pool in pools:
        cid = pool["category_id"]
        for image_id in pool["positives"]:
            large = [annotation for annotation in annotations.get(image_id, [])
                     if annotation["category_id"] == cid and not annotation["iscrowd"]
                     and annotation["bbox"][2] * annotation["bbox"][3] >= MIN_AREA_RATIO * image_area[image_id]]
            if not large:
                violations.append(f"P {image_id}: {pool['name']}의 큰 bbox가 없음")
        for image_id in pool["negatives"]:
            if any(annotation["category_id"] == cid for annotation in annotations.get(image_id, [])):
                violations.append(f"N {image_id}: {pool['name']} 주석이 있음")
    return violations


def shared_within_pool(pools: list[dict]) -> dict[int, list[str]]:
    """한 풀 안에서 두 클래스 이상이 쓰는 이미지: {image id: ["P person", "N dog", ...]}

    한 클래스의 P와 N은 정의상 겹치지 않으므로, 쓰임이 둘 이상이면 서로 다른 클래스다.
    """
    uses = {}
    for pool in pools:
        for role, key in [("P", "positives"), ("N", "negatives")]:
            for image_id in pool[key]:
                uses.setdefault(image_id, []).append(f"{role} {pool['name']}")
    return {image_id: roles for image_id, roles in uses.items() if len(roles) > 1}


def check_new_pools(coco: dict, base_pools: list[dict], built: dict[str, list[dict]]) -> None:
    """파일을 쓰기 전의 점검. 하나라도 어기면 모든 문제를 담아 ValueError를 낸다

    기존 풀과 겹치는 이미지 0, 새 풀끼리 겹치는 이미지 0, 클래스 id와 순서가 기존과 같음,
    클래스마다 서로 다른 P와 N이 POOL_SIZE장씩, 주석의 정의를 지킴 (annotation_violations)
    """
    problems = []
    base_images = pool_images(base_pools)
    base_classes = [pool["category_id"] for pool in base_pools]
    for name, pools in built.items():
        overlap = pool_images(pools) & base_images
        if overlap:
            problems.append(f"{name}: 기존 풀과 겹치는 이미지 {len(overlap)}장")
        if [pool["category_id"] for pool in pools] != base_classes:
            problems.append(f"{name}: 클래스 id나 순서가 기존 풀과 다름")
        for pool in pools:
            if len(set(pool["positives"])) != POOL_SIZE or len(set(pool["negatives"])) != POOL_SIZE:
                problems.append(f"{name}: {pool['name']}의 P 또는 N이 서로 다른 {POOL_SIZE}장이 아님")
        problems += [f"{name}: {violation}" for violation in annotation_violations(coco, pools)]
    for first, second in itertools.combinations(built, 2):
        overlap = pool_images(built[first]) & pool_images(built[second])
        if overlap:
            problems.append(f"{first}와 {second}: 겹치는 이미지 {len(overlap)}장")
    if problems:
        raise ValueError("새 풀의 점검 실패:\n" + "\n".join(problems))


def pool_files(pools: list[dict], seed_prefix: str) -> dict[str, str]:
    """풀 하나에서 만드는 파일 세 개의 내용: {파일 이름: 텍스트}

    형식은 기존 build_pools.py, build_pairs.py, build_cases.py가 쓰는 파일과 같다. pair의 seed 접두어는 seed_prefix다.
    """
    pairs = [pair for pool in pools for pair in make_pairs(pool, seed_prefix)]
    cases = [case for pair in pairs for case in make_cases(pair)]
    return {
        "pools.json": json.dumps(pools, indent=2),
        "pairs.json": json.dumps(pairs, indent=2),
        "cases.jsonl": "".join(json.dumps(case) + "\n" for case in cases),
    }


def sha256_text(text: str) -> str:
    """텍스트를 UTF-8로 쓴 파일의 SHA-256"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_summary(built: dict[str, list[dict]], files: dict[str, dict[str, str]], git: dict,
                 sources: dict[str, str]) -> dict:
    """생성 기록 (비공개, data/ 아래): 환경, 입력 파일의 해시, 풀마다 파일 해시와 후보 수와 풀 안 공유 이미지

    git: runner.git_state()의 값. sources: {입력 파일 이름: SHA-256}. 점검은 통과한 뒤에만 이 기록을 만든다.
    """
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "procedure": "docs/preregistration.md 2.2 (Stage 1, commit 1382ac5)",
        "python": platform.python_version(),
        "git": git,
        "sources": sources,
        "checks": "passed: no overlap with the base pools or between new pools, classes as in the base pools, "
                  f"{POOL_SIZE} P and {POOL_SIZE} N per class, annotation definitions hold",
        "pools": {
            name: {
                "seed_prefix": f"{SEED}:{name}",
                "files_sha256": {file_name: sha256_text(text) for file_name, text in files[name].items()},
                "n_images": len(pool_images(pools)),
                "candidates_after_exclusion": {
                    pool["name"]: [pool["n_positive_candidates"], pool["n_negative_candidates"]] for pool in pools
                },
                "shared_within_pool": {str(image_id): roles for image_id, roles in shared_within_pool(pools).items()},
            }
            for name, pools in built.items()
        },
    }


def existing_outputs(cases_dir: Path, names: list[str]) -> list[Path]:
    """이미 있는 출력: 풀 폴더, 요약 파일"""
    return [path for path in [*(cases_dir / name for name in names), cases_dir / SUMMARY_NAME] if path.exists()]


def write_new_pools(cases_dir: Path, files: dict[str, dict[str, str]], summary: dict) -> None:
    """파일을 쓴다. 출력이 하나라도 이미 있으면 아무것도 쓰지 않고 FileExistsError를 낸다

    임시 폴더에 풀 폴더를 모두 쓴 뒤 하나씩 제자리로 옮기고, 요약은 마지막에 쓴다.
    중간에 실패하면 옮긴 폴더와 요약을 지우고 에러를 다시 낸다. 임시 폴더는 성공해도 실패해도 지운다.
    """
    existing = existing_outputs(cases_dir, list(files))
    if existing:
        raise FileExistsError("이미 있는 출력이 있어 쓰지 않는다: " + ", ".join(str(path) for path in existing))
    temporary = cases_dir / TEMPORARY_NAME
    shutil.rmtree(temporary, ignore_errors=True)  # 예전 실행이 남긴 임시 폴더
    moved = []
    try:
        for name, contents in files.items():
            (temporary / name).mkdir(parents=True)
            for file_name, text in contents.items():
                (temporary / name / file_name).write_text(text, encoding="utf-8")
        for name in files:
            (temporary / name).rename(cases_dir / name)
            moved.append(cases_dir / name)
        (cases_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    except BaseException:
        for path in moved:
            shutil.rmtree(path, ignore_errors=True)
        (cases_dir / SUMMARY_NAME).unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def sha256_file(path: Path) -> str:
    """파일의 SHA-256"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    state = git_state()
    if state["dirty"]:
        raise SystemExit("커밋하지 않은 변경이 있어 멈춘다. 풀을 만드는 코드는 먼저 커밋해야 한다 (사전 등록 2.2절).")
    existing = existing_outputs(CASES_DIR, POOL_NAMES)
    if existing:
        raise SystemExit("이미 있는 출력이 있어 멈춘다: " + ", ".join(str(path) for path in existing))

    with open(INSTANCES_JSON) as f:
        coco = json.load(f)
    with open(POOLS_JSON) as f:
        base_pools = json.load(f)
    built = build_new_pools(coco, base_pools)
    check_new_pools(coco, base_pools, built)
    files = {name: pool_files(pools, f"{SEED}:{name}") for name, pools in built.items()}
    sources = {INSTANCES_JSON.name: sha256_file(INSTANCES_JSON), "pools.json (기존)": sha256_file(POOLS_JSON)}
    summary = make_summary(built, files, state, sources)
    write_new_pools(CASES_DIR, files, summary)

    print(f"[ok] 점검 통과. 코드 {state['commit'][:7]}, Python {summary['python']}")
    for name, record in summary["pools"].items():
        print(f"  {name}: 클래스 {len(built[name])}개, 이미지 {record['n_images']}장, "
              f"풀 안에서 여러 클래스가 쓰는 이미지 {len(record['shared_within_pool'])}장")
        for file_name, digest in record["files_sha256"].items():
            print(f"    {file_name}  {digest}")
    print(f"[ok] -> {CASES_DIR}/{{{','.join(POOL_NAMES)}}}/, {CASES_DIR / SUMMARY_NAME}")


if __name__ == "__main__":
    main()
