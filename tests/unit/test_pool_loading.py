"""
풀을 고르는 데이터 읽기와 데이터 출처 확인 단위 테스트 (가짜 풀 파일만 쓴다. 실제 데이터와 모델을 쓰지 않는다)
pytest tests/unit/test_pool_loading.py -v

mivlm.config.CASES_DIR를 임시 폴더로 바꿔 끼운다. pool_dir가 부를 때마다 CASES_DIR를 읽으므로 다른 모듈에도 반영된다.
"""
import argparse
import hashlib
import json

import pytest

import mivlm.config
import mivlm.evaluation.decomposition
from mivlm.config import pool_dir
from mivlm.data.cases import (
    data_sha256, load_cases, load_pools, load_run_cases, pool_file, run_pool, shared_run_pool,
)
from mivlm.evaluation.decomposition import load_case_rows
from mivlm.evaluation.metrics import load_rows
from mivlm.runner import check_or_save_meta, data_meta, parse_run_args, run_name


def _case(case_id: str, name: str) -> dict:
    return {"case_id": case_id, "group_id": f"group_01_{name}", "pair_id": 1, "object": name, "variant": "PPP",
            "image_ids": [1, 2, 3], "label": "Yes"}


BASE_CASES = [_case("base_b", "cat"), _case("base_a", "cat")]  # 파일 순서가 id 순서와 다르다
DEV_CASES = [_case("dev_1", "dog"), _case("dev_2", "dog"), _case("dev_3", "dog")]


def _write(folder, cases: list[dict], pools: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "cases.jsonl").write_text("".join(json.dumps(case) + "\n" for case in cases))
    (folder / "pools.json").write_text(json.dumps(pools, indent=2))


@pytest.fixture
def cases_dir(tmp_path, monkeypatch):
    """기존 탐색 데이터와 dev 풀이 있는 임시 data/cases. confirm, eval 폴더는 없다"""
    root = tmp_path / "cases"
    _write(root, BASE_CASES, [{"name": "cat"}])
    _write(root / "dev", DEV_CASES, [{"name": "dog"}])
    monkeypatch.setattr(mivlm.config, "CASES_DIR", root)
    return root


def _run_dir(tmp_path, name: str, meta: dict, results: list[dict] | None = None):
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps(meta))
    (run_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in results or []))
    return run_dir


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- pool_dir, 읽기


def test_pool_dir_maps_none_to_the_exploration_data_and_names_to_subfolders(cases_dir):
    assert pool_dir(None) == cases_dir
    assert pool_dir("dev") == cases_dir / "dev"
    assert pool_dir("eval") == cases_dir / "eval"


@pytest.mark.parametrize("name", ["x", "../dev", "", "DEV", "dev/", 1])
def test_pool_dir_rejects_unknown_names(cases_dir, name):
    with pytest.raises(ValueError, match="알 수 없는 풀 이름"):
        pool_dir(name)


def test_load_cases_reads_each_pool_in_file_order_with_fields_unchanged(cases_dir):
    assert load_cases() == BASE_CASES  # 기존 탐색 데이터: 인자 없는 호출은 예전과 같다
    assert load_cases(None) == BASE_CASES
    assert load_cases("dev") == DEV_CASES
    assert load_pools() == [{"name": "cat"}] and load_pools("dev") == [{"name": "dog"}]


def test_a_missing_pool_file_fails_with_the_pool_name(cases_dir):
    with pytest.raises(FileNotFoundError, match="풀 eval의 cases.jsonl이 없음"):
        load_cases("eval")
    (cases_dir / "pools.json").unlink()
    with pytest.raises(FileNotFoundError, match="기존 탐색 데이터의 pools.json이 없음"):
        load_pools()


def test_data_sha256_hashes_the_file_bytes(cases_dir):
    assert data_sha256("dev") == {"cases_sha256": _sha(cases_dir / "dev" / "cases.jsonl"),
                                  "pools_sha256": _sha(cases_dir / "dev" / "pools.json")}
    assert pool_file(None, "cases.jsonl") == cases_dir / "cases.jsonl"


# ---------------------------------------------------------------- 실행 폴더의 풀 (run_pool)


def test_a_run_without_a_pool_key_uses_the_exploration_data(cases_dir, tmp_path):
    run_dir = _run_dir(tmp_path, "old", {"pilot": False})  # 풀이 생기기 전의 실행, 해시도 없다

    assert run_pool(run_dir) is None
    assert load_run_cases(run_dir) == BASE_CASES


def test_a_run_with_a_pool_and_matching_hashes_uses_that_pool(cases_dir, tmp_path):
    run_dir = _run_dir(tmp_path, "dev", {"pilot": False, **data_meta("dev")})

    assert run_pool(run_dir) == "dev"
    assert load_run_cases(run_dir) == DEV_CASES


@pytest.mark.parametrize("pool", ["x", None, "../dev"])
def test_an_invalid_pool_name_in_meta_is_an_error_not_the_exploration_data(cases_dir, tmp_path, pool):
    run_dir = _run_dir(tmp_path, "bad", {"pilot": False, "pool": pool})

    with pytest.raises(ValueError, match="알 수 없는 풀 이름"):
        run_pool(run_dir)


def test_a_hash_that_differs_from_the_current_file_is_an_error(cases_dir, tmp_path):
    run_dir = _run_dir(tmp_path, "dev", {"pilot": False, **data_meta("dev")})
    (cases_dir / "dev" / "cases.jsonl").write_text(json.dumps(DEV_CASES[0]) + "\n")  # 실행 뒤에 파일이 바뀌었다

    with pytest.raises(ValueError, match=r"실행 당시 데이터와 지금 풀 dev의 파일이 다름: \['cases_sha256'\]"):
        run_pool(run_dir)


def test_load_rows_joins_the_results_with_the_cases_of_the_run_pool(cases_dir, tmp_path):
    results = [{"case_id": case["case_id"], "prediction": "Yes", "logit_margin": 1.0} for case in reversed(DEV_CASES)]
    run_dir = _run_dir(tmp_path, "dev", {"pilot": False, **data_meta("dev")}, results)

    rows = load_rows(run_dir)

    assert [row["case_id"] for row in rows] == ["dev_1", "dev_2", "dev_3"]  # 결과 순서가 아니라 케이스 순서, id로 붙인다
    assert rows[0]["object"] == "dog" and rows[0]["prediction"] == "Yes"


# ---------------------------------------------------------------- A, B, C의 같은 출처 (shared_run_pool)


def test_runs_on_the_same_pool_and_data_share_that_pool(cases_dir, tmp_path):
    runs = [_run_dir(tmp_path, name, {"pilot": False, **data_meta("dev")}) for name in "abc"]

    assert shared_run_pool(runs) == "dev"


def test_runs_on_different_pools_are_an_error(cases_dir, tmp_path):
    runs = [_run_dir(tmp_path, "a", {"pilot": False, **data_meta("dev")}),
            _run_dir(tmp_path, "b", {"pilot": False, **data_meta(None)})]

    with pytest.raises(ValueError, match="실행마다 풀이 다름"):
        shared_run_pool(runs)


def test_runs_with_the_same_pool_but_a_different_case_hash_are_an_error(cases_dir, tmp_path):
    runs = [_run_dir(tmp_path, "a", {"pilot": False, **data_meta("dev")}),
            _run_dir(tmp_path, "b", {"pilot": False, **data_meta("dev"), "cases_sha256": "0" * 64})]

    with pytest.raises(ValueError, match="cases_sha256"):
        shared_run_pool(runs)


def test_load_case_rows_refuses_a_b_c_runs_from_different_pools(cases_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(mivlm.evaluation.decomposition, "PREDICTIONS_DIR", tmp_path / "predictions")
    for folder, pool in [("01_baseline", "dev"), ("02_decomposition/b_single_image", None),
                         ("02_decomposition/c_per_image_answers", "dev")]:
        run_dir = tmp_path / "predictions" / folder / "m_dev"
        run_dir.mkdir(parents=True)
        (run_dir / "meta.json").write_text(json.dumps({"pilot": False, **data_meta(pool)}))

    with pytest.raises(ValueError, match="실행마다 풀이 다름"):
        load_case_rows("m_dev")


# ---------------------------------------------------------------- runner: 옵션, 폴더 이름, meta


def test_without_pool_options_the_run_uses_the_exploration_data():
    args = parse_run_args("test", [])

    assert (args.pool, args.unseal) == (None, None)


@pytest.mark.parametrize("argv", [["--pool", "dev"], ["--pool", "confirm", "--unseal", "confirm"],
                                  ["--pool", "eval", "--unseal", "eval"]])
def test_allowed_pool_options(argv):
    args = parse_run_args("test", argv)

    assert args.pool == argv[1]


@pytest.mark.parametrize("argv", [
    ["--pool", "eval"],                          # 봉인된 풀인데 --unseal이 없다
    ["--pool", "confirm"],
    ["--pool", "confirm", "--unseal", "eval"],   # 다른 이름으로 풀었다
    ["--pool", "dev", "--unseal", "confirm"],    # 봉인되지 않은 풀에 --unseal
    ["--unseal", "eval"],                        # 풀 없이 --unseal
    ["--pool", "x"],                             # 등록되지 않은 풀
])
def test_refused_pool_options(argv):
    with pytest.raises(SystemExit):
        parse_run_args("test", argv)


def test_run_name_adds_the_pool_after_the_dtype_and_before_pilot():
    def name(**options):
        return run_name(argparse.Namespace(**{"model": "m", "dtype": None, "pool": None, "pilot": False, **options}))

    assert name() == "m"  # 기존 탐색 데이터: 예전 이름 그대로
    assert name(pool="dev") == "m_dev"
    assert name(dtype="float32", pool="dev", pilot=True) == "m_float32_dev_pilot"


def test_data_meta_records_the_pool_only_for_new_pools(cases_dir):
    assert data_meta(None) == {"cases_sha256": _sha(cases_dir / "cases.jsonl"),
                               "pools_sha256": _sha(cases_dir / "pools.json")}
    assert data_meta("dev") == {"pool": "dev", "cases_sha256": _sha(cases_dir / "dev" / "cases.jsonl"),
                                "pools_sha256": _sha(cases_dir / "dev" / "pools.json")}


def test_resuming_with_another_pool_or_data_is_refused(cases_dir, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    meta = {"model": {"name": "m", "dtype": "bf16"}, "pilot": False, "question": "q", **data_meta("dev")}
    check_or_save_meta(run_dir, meta)

    check_or_save_meta(run_dir, dict(meta))  # 같은 조건이면 이어서 돌릴 수 있다
    for changed in [{**meta, "pool": "confirm"}, {key: value for key, value in meta.items() if key != "pool"},
                    {**meta, "cases_sha256": "0" * 64}, {**meta, "pilot": True}, {**meta, "question": "other"}]:
        with pytest.raises(RuntimeError, match="기존 실행과 조건이 다름"):
            check_or_save_meta(run_dir, changed)
