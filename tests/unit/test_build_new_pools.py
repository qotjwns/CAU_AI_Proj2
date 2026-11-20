"""
scripts/build_new_pools.py 단위 테스트 (작은 가상 COCO 주석. 실제 데이터와 모델을 쓰지 않고, 새 풀을 만들지 않는다)
pytest tests/unit/test_build_new_pools.py -v
"""
import hashlib
import json
import platform
import random

import pytest

from build_new_pools import (
    POOL_NAMES, SUMMARY_NAME, annotation_violations, build_new_pools, check_new_pools, make_summary, pool_files,
    pool_images, sample_pool, shared_within_pool, write_new_pools,
)
from build_pools import find_candidates

CAT, DOG = 1, 2


def _coco() -> dict:
    """이미지 402장 (100x100). 1~80은 큰 cat, 81~160은 큰 dog, 161~400은 주석 없음.
    401은 작은 cat bbox(1%)만, 402는 crowd인 큰 cat만 있어서 cat의 P도 N도 아니다.
    후보를 넉넉히 두어, 풀 세 개를 차례로 뽑아도 모자라지 않게 한다"""
    images = [{"id": image_id, "width": 100, "height": 100} for image_id in range(1, 403)]
    annotations = (
        [{"image_id": i, "category_id": CAT, "bbox": [0, 0, 50, 50], "iscrowd": 0} for i in range(1, 81)]
        + [{"image_id": i, "category_id": DOG, "bbox": [0, 0, 60, 60], "iscrowd": 0} for i in range(81, 161)]
        + [{"image_id": 401, "category_id": CAT, "bbox": [0, 0, 10, 10], "iscrowd": 0},
           {"image_id": 402, "category_id": CAT, "bbox": [0, 0, 80, 80], "iscrowd": 1}]
    )
    return {"images": images, "annotations": annotations,
            "categories": [{"id": CAT, "name": "cat"}, {"id": DOG, "name": "dog"}]}


# 기존 풀 (기존 311장의 역할). cat의 N 일부는 dog의 P 이미지다
BASE_POOLS = [
    {"category_id": CAT, "name": "cat", "positives": list(range(1, 9)), "negatives": list(range(81, 89))},
    {"category_id": DOG, "name": "dog", "positives": list(range(89, 97)), "negatives": list(range(161, 169))},
]
CLASSES = [(CAT, "cat"), (DOG, "dog")]


# ---------------------------------------------------------------- pool_images, sample_pool


def test_pool_images_collects_both_roles_of_every_class():
    assert pool_images(BASE_POOLS) == set(range(1, 9)) | set(range(81, 97)) | set(range(161, 169))


def test_candidates_follow_the_existing_definitions():
    positives, negatives = find_candidates(_coco())

    assert positives[CAT] == list(range(1, 81))  # 401(작은 bbox)과 402(crowd)는 P가 아니다
    assert 401 not in negatives[CAT] and 402 not in negatives[CAT]  # cat 주석이 있으니 N도 아니다
    assert negatives[CAT] == list(range(81, 401))


def test_sample_pool_uses_the_registered_seed_strings():
    positives, negatives = find_candidates(_coco())
    excluded = pool_images(BASE_POOLS)

    pools = sample_pool("42:dev", CLASSES, positives, negatives, excluded)

    cat_p = [i for i in positives[CAT] if i not in excluded]
    cat_n = [i for i in negatives[CAT] if i not in excluded]
    assert pools[0]["positives"] == random.Random(f"42:dev:{CAT}:positive_candidates").sample(cat_p, 8)
    assert pools[0]["negatives"] == random.Random(f"42:dev:{CAT}:negative_candidates").sample(cat_n, 8)
    assert (pools[0]["n_positive_candidates"], pools[0]["n_negative_candidates"]) == (len(cat_p), len(cat_n))
    assert list(pools[0]) == ["category_id", "name", "n_positive_candidates", "n_negative_candidates",
                              "positives", "negatives"]  # 기존 pools.json과 같은 키 순서


def test_prefix_42_without_exclusion_is_the_original_procedure():
    positives, negatives = find_candidates(_coco())

    pools = sample_pool("42", CLASSES, positives, negatives, set())

    # 기존 build_pools.py의 main이 쓰는 seed 문자열과 같다
    assert pools[1]["positives"] == random.Random(f"42:{DOG}:positive_candidates").sample(positives[DOG], 8)
    assert pools[1]["negatives"] == random.Random(f"42:{DOG}:negative_candidates").sample(negatives[DOG], 8)


def test_too_few_candidates_stop_with_the_class_and_the_shortage():
    positives, negatives = find_candidates(_coco())
    excluded = set(range(1, 76))  # cat의 P 후보가 76~80의 5장만 남는다

    with pytest.raises(ValueError, match=r"cat: P 5장 \(모자람 3\)"):
        sample_pool("42:dev", CLASSES, positives, negatives, excluded)


# ---------------------------------------------------------------- build_new_pools


def test_new_pools_avoid_the_base_images_and_each_other():
    built = build_new_pools(_coco(), BASE_POOLS)

    assert list(built) == POOL_NAMES == ["dev", "confirm", "eval"]
    for pools in built.values():
        assert not pool_images(pools) & pool_images(BASE_POOLS)
    assert not pool_images(built["dev"]) & pool_images(built["confirm"])
    assert not pool_images(built["dev"]) & pool_images(built["eval"])
    assert not pool_images(built["confirm"]) & pool_images(built["eval"])


def test_each_pool_samples_from_what_the_earlier_pools_left():
    coco = _coco()
    built = build_new_pools(coco, BASE_POOLS)
    positives, negatives = find_candidates(coco)

    excluded = pool_images(BASE_POOLS) | pool_images(built["dev"])
    assert built["confirm"] == sample_pool("42:confirm", CLASSES, positives, negatives, excluded)
    excluded |= pool_images(built["confirm"])
    assert built["eval"] == sample_pool("42:eval", CLASSES, positives, negatives, excluded)


def test_the_same_inputs_give_the_same_pools():
    assert build_new_pools(_coco(), BASE_POOLS) == build_new_pools(_coco(), BASE_POOLS)


def test_classes_keep_the_base_ids_and_order():
    built = build_new_pools(_coco(), list(reversed(BASE_POOLS)))

    assert [pool["name"] for pool in built["dev"]] == ["dog", "cat"]


# ---------------------------------------------------------------- 점검


def test_annotation_violations_find_images_that_break_the_definitions():
    coco = _coco()
    pools = [{"category_id": CAT, "name": "cat", "positives": [1, 401, 402], "negatives": [300, 3]}]

    violations = annotation_violations(coco, pools)

    # 401은 작은 bbox, 402는 crowd라 P가 아니다. 3은 cat 주석이 있어 N이 아니다
    assert violations == ["P 401: cat의 큰 bbox가 없음", "P 402: cat의 큰 bbox가 없음", "N 3: cat 주석이 있음"]


def test_built_pools_pass_the_checks():
    coco = _coco()

    check_new_pools(coco, BASE_POOLS, build_new_pools(coco, BASE_POOLS))  # 에러가 나지 않아야 한다


def test_the_checks_report_every_problem():
    coco = _coco()
    built = build_new_pools(coco, BASE_POOLS)
    built["confirm"][0]["positives"][0] = built["dev"][0]["positives"][0]  # 새 풀끼리 겹치게 한다
    built["eval"][1]["negatives"][0] = 1  # 기존 풀의 이미지 (dog 주석은 없어서 주석 조건은 지킨다)

    with pytest.raises(ValueError) as error:
        check_new_pools(coco, BASE_POOLS, built)

    message = str(error.value)
    assert "dev와 confirm: 겹치는 이미지 1장" in message
    assert "eval: 기존 풀과 겹치는 이미지 1장" in message


def test_shared_within_pool_lists_images_used_by_two_classes():
    pools = [{"name": "cat", "positives": [1, 2], "negatives": [9]},
             {"name": "dog", "positives": [9], "negatives": [1]}]

    assert shared_within_pool(pools) == {1: ["P cat", "N dog"], 9: ["N cat", "P dog"]}


# ---------------------------------------------------------------- 파일과 기록


def test_pool_files_use_the_original_formats_and_the_pool_seed_for_pairs():
    pools = build_new_pools(_coco(), BASE_POOLS)["dev"]

    files = pool_files(pools, "42:dev")

    assert files["pools.json"] == json.dumps(pools, indent=2)
    pairs = json.loads(files["pairs.json"])
    assert len(pairs) == 20 * len(pools)
    cases = [json.loads(line) for line in files["cases.jsonl"].splitlines()]
    assert len(cases) == 6 * len(pairs)
    # pair의 N은 풀 이름이 붙은 seed로 뽑는다
    first = pairs[0]
    expected = random.Random(f"42:dev:{first['group_id']}:pairs").sample(pools[0]["negatives"], 3)
    assert first["n"] == expected


def test_make_summary_records_the_environment_and_the_file_hashes():
    built = build_new_pools(_coco(), BASE_POOLS)
    files = {name: pool_files(pools, f"42:{name}") for name, pools in built.items()}
    git = {"commit": "abc", "dirty": False}

    summary = make_summary(built, files, git, {"instances": "f00"})

    assert summary["python"] == platform.python_version()
    assert summary["git"] == git and summary["sources"] == {"instances": "f00"}
    dev = summary["pools"]["dev"]
    assert dev["seed_prefix"] == "42:dev"
    assert dev["files_sha256"]["cases.jsonl"] == hashlib.sha256(files["dev"]["cases.jsonl"].encode()).hexdigest()
    assert dev["candidates_after_exclusion"]["cat"] == [built["dev"][0]["n_positive_candidates"],
                                                        built["dev"][0]["n_negative_candidates"]]


def _files() -> dict[str, dict[str, str]]:
    return {name: {"pools.json": f"[{index}]", "pairs.json": "[]", "cases.jsonl": ""}
            for index, name in enumerate(POOL_NAMES)}


def test_write_new_pools_writes_every_file_and_leaves_no_temporary_folder(tmp_path):
    write_new_pools(tmp_path, _files(), {"ok": True})

    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([*POOL_NAMES, SUMMARY_NAME])
    assert (tmp_path / "confirm" / "pools.json").read_text() == "[1]"
    assert json.loads((tmp_path / SUMMARY_NAME).read_text()) == {"ok": True}


def test_write_new_pools_writes_nothing_when_an_output_exists(tmp_path):
    (tmp_path / "eval").mkdir()

    with pytest.raises(FileExistsError):
        write_new_pools(tmp_path, _files(), {"ok": True})

    assert [path.name for path in tmp_path.iterdir()] == ["eval"]  # 원래 있던 폴더만 남는다
    assert list((tmp_path / "eval").iterdir()) == []


def test_write_new_pools_removes_partial_results_on_failure(tmp_path):
    # 요약은 마지막에 쓴다. 요약을 JSON으로 만들 수 없으면 이미 옮긴 풀 폴더까지 지우고 에러를 다시 낸다
    with pytest.raises(TypeError):
        write_new_pools(tmp_path, _files(), {"not json": {1, 2}})

    assert list(tmp_path.iterdir()) == []
