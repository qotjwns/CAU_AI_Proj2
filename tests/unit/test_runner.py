"""
src/mivlm/runner.py 단위 테스트. 모델 대신 가짜 추론 함수를 쓴다
pytest tests/unit/test_runner.py -v
"""
import json

import pytest

from mivlm.runner import check_or_save_meta, run_items

ITEMS = [{"case_id": "a", "n": 1}, {"case_id": "b", "n": 2}, {"case_id": "c", "n": 3}]


def _fake_infer(item):
    return {"prediction": "Yes", "doubled": item["n"] * 2}


def _read_results(run_dir):
    with open(run_dir / "results.jsonl") as f:
        return [json.loads(line) for line in f]


# ------------------------------------------------------------------- run_items


def test_first_run_infers_every_item(tmp_path):
    assert run_items(tmp_path, ITEMS, "case_id", _fake_infer) == 3
    assert _read_results(tmp_path) == [
        {"case_id": "a", "prediction": "Yes", "doubled": 2},
        {"case_id": "b", "prediction": "Yes", "doubled": 4},
        {"case_id": "c", "prediction": "Yes", "doubled": 6},
    ]


def test_second_run_skips_everything(tmp_path):
    run_items(tmp_path, ITEMS, "case_id", _fake_infer)

    assert run_items(tmp_path, ITEMS, "case_id", _fake_infer) == 0
    assert len(_read_results(tmp_path)) == 3


def test_resume_infers_only_missing_items(tmp_path):
    # 'a'까지만 끝난 상태에서 끊긴 실행을 흉내 낸다
    (tmp_path / "results.jsonl").write_text(json.dumps({"case_id": "a", "prediction": "No"}) + "\n")
    called = []

    def infer_and_record(item):
        called.append(item["case_id"])
        return _fake_infer(item)

    assert run_items(tmp_path, ITEMS, "case_id", infer_and_record) == 2
    assert called == ["b", "c"]
    assert [row["case_id"] for row in _read_results(tmp_path)] == ["a", "b", "c"]
    assert _read_results(tmp_path)[0]["prediction"] == "No"  # 이미 있던 줄은 그대로 둔다


# ---------------------------------------------------------- check_or_save_meta

META = {"started_at": "t1", "command": "run.py", "git": {"commit": "aaa"}, "pilot": False, "model": {"dtype": "bfloat16"}}


def test_meta_is_saved_when_absent(tmp_path):
    check_or_save_meta(tmp_path, META)

    assert json.loads((tmp_path / "meta.json").read_text()) == META


def test_same_conditions_pass_and_keep_first_meta(tmp_path):
    check_or_save_meta(tmp_path, META)

    # 시작 시각, 명령, git은 달라도 된다
    check_or_save_meta(tmp_path, {**META, "started_at": "t2", "command": "./run.py", "git": {"commit": "bbb"}})

    assert json.loads((tmp_path / "meta.json").read_text())["started_at"] == "t1"


def test_changed_condition_raises(tmp_path):
    check_or_save_meta(tmp_path, META)

    with pytest.raises(RuntimeError):
        check_or_save_meta(tmp_path, {**META, "model": {"dtype": "float32"}})
