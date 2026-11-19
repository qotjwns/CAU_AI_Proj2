"""
src/mivlm/run_files.py 단위 테스트
pytest tests/unit/test_run_files.py -v
"""
import json

from mivlm.run_files import read_meta, read_results


def test_read_meta_returns_the_saved_conditions(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"experiment": "01_baseline", "pilot": True}))

    assert read_meta(tmp_path) == {"experiment": "01_baseline", "pilot": True}


def test_read_results_returns_one_dict_per_line_in_file_order(tmp_path):
    (tmp_path / "results.jsonl").write_text(
        '{"case_id": "b", "prediction": "No"}\n'
        '{"case_id": "a", "prediction": "Yes"}\n'
    )

    assert read_results(tmp_path) == [{"case_id": "b", "prediction": "No"}, {"case_id": "a", "prediction": "Yes"}]
