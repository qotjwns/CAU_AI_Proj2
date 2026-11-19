"""
src/mivlm/analysis/pair_store.py 단위 테스트 (모델은 로딩하지 않는다)
pytest tests/unit/test_pair_store.py -v
"""
import hashlib
import json

import pytest
import torch

from mivlm.analysis.pair_store import check_saved_pairs, save_pair_record


def _record(pair: str = "fake_pair", value: float = 1.0) -> dict:
    """실제 기록과 같은 종류의 값(글자, 숫자, bool, 목록, 텐서)만 담은 작은 기록"""
    return {
        "pair": pair,
        "cases": {"PPP": {"case_id": f"{pair}_PPP", "token_spans": [[2, 4]], "logit_margin": 2.25,
                          "pre_answer_hidden": torch.full((2, 3), value)}},
        "comparisons": [{"family": "same_prefix", "same_token_index": True, "token_cosine": torch.ones(2, 4)}],
    }


def _finish(run_dir, record: dict) -> dict:
    """run_items가 하는 일을 흉내 낸다: 저장한 뒤 그 정보를 results.jsonl에 한 줄로 쓴다"""
    saved = save_pair_record(run_dir, record)
    with open(run_dir / "results.jsonl", "a") as f:
        f.write(json.dumps({"pair": record["pair"], **saved}) + "\n")
    return saved


# ----------------------------------------------------------- save_pair_record


def test_save_writes_one_file_per_pair_and_returns_its_name_and_hash(tmp_path):
    saved = save_pair_record(tmp_path, _record())

    path = tmp_path / "pairs" / "fake_pair.pt"
    assert saved == {"file": "pairs/fake_pair.pt", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_save_leaves_no_temporary_file(tmp_path):
    save_pair_record(tmp_path, _record())

    assert [path.name for path in (tmp_path / "pairs").iterdir()] == ["fake_pair.pt"]


def test_saved_record_reads_back_the_same(tmp_path):
    record = _record()
    saved = save_pair_record(tmp_path, record)

    loaded = torch.load(tmp_path / saved["file"], map_location="cpu", weights_only=True)

    assert loaded["pair"] == "fake_pair"
    assert loaded["cases"]["PPP"]["token_spans"] == [[2, 4]]
    assert loaded["cases"]["PPP"]["logit_margin"] == 2.25
    assert loaded["comparisons"][0]["same_token_index"] is True
    assert torch.equal(loaded["cases"]["PPP"]["pre_answer_hidden"], record["cases"]["PPP"]["pre_answer_hidden"])
    assert loaded["comparisons"][0]["token_cosine"].dtype == torch.float32


def test_saving_the_same_pair_again_replaces_the_file(tmp_path):
    first = save_pair_record(tmp_path, _record(value=1.0))
    second = save_pair_record(tmp_path, _record(value=2.0))

    loaded = torch.load(tmp_path / second["file"], map_location="cpu", weights_only=True)
    assert first["sha256"] != second["sha256"]
    assert loaded["cases"]["PPP"]["pre_answer_hidden"].tolist() == [[2.0] * 3] * 2


# ---------------------------------------------------------- check_saved_pairs


def test_check_passes_before_anything_is_saved(tmp_path):
    check_saved_pairs(tmp_path)  # 폴더가 비어 있어도 에러가 없어야 한다


def test_check_passes_when_every_finished_pair_has_its_file(tmp_path):
    _finish(tmp_path, _record("pair_a"))
    _finish(tmp_path, _record("pair_b"))

    check_saved_pairs(tmp_path)


def test_check_fails_when_a_finished_pair_has_no_file(tmp_path):
    saved = _finish(tmp_path, _record())
    (tmp_path / saved["file"]).unlink()

    with pytest.raises(RuntimeError, match="fake_pair"):
        check_saved_pairs(tmp_path)


def test_check_fails_when_a_file_changed_after_saving(tmp_path):
    saved = _finish(tmp_path, _record())
    with open(tmp_path / saved["file"], "ab") as f:
        f.write(b"x")

    with pytest.raises(RuntimeError, match="내용이 저장할 때와 다름"):
        check_saved_pairs(tmp_path)


def test_check_removes_temporary_files_left_by_an_interrupted_save(tmp_path):
    _finish(tmp_path, _record("pair_a"))
    leftover = tmp_path / "pairs" / "pair_b.pt.tmp"
    leftover.write_bytes(b"half written")

    check_saved_pairs(tmp_path)

    assert not leftover.exists()
    assert (tmp_path / "pairs" / "pair_a.pt").exists()


def test_check_accepts_a_file_whose_pair_is_not_in_results_yet(tmp_path):
    # 이름을 바꾼 직후, results.jsonl에 쓰기 전에 끊긴 경우. 그 pair는 다시 실행되어 새 파일로 바뀐다
    _finish(tmp_path, _record("pair_a"))
    save_pair_record(tmp_path, _record("pair_b"))

    check_saved_pairs(tmp_path)

    assert (tmp_path / "pairs" / "pair_b.pt").exists()
