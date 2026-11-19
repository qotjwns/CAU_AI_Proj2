"""
src/mivlm/evaluation/report.py의 표 출력 단위 테스트
pytest tests/unit/test_report.py -v
"""
import torch

from mivlm.evaluation.report import interval_text, print_class_table, print_layer_table


def test_layer_table_has_one_row_per_layer_and_one_column_per_name(capsys):
    print_layer_table("제목", {1: torch.tensor([1.0, 0.5]), "2*": torch.tensor([0.25, 0.125])}, ".3f")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ""
    assert lines[1] == "[제목]"
    assert lines[2].split() == ["layer", "1", "2*"]
    assert lines[3].split() == ["0", "1.000", "0.250"]
    assert lines[4].split() == ["1", "0.500", "0.125"]
    assert len(lines) == 5


def test_class_table_has_one_row_per_class_and_formats_each_column(capsys):
    table = {"cat": {"n": 20, "rate": 0.25, "drop": -1.5}, "dining table": {"n": 5, "rate": 1.0, "drop": 0.125}}
    print_class_table("제목", table, {"n": ("n", "d"), "yes": ("rate", ".0%"), "dm": ("drop", "+.2f")})

    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "[제목]"
    assert lines[2].split() == ["n", "yes", "dm"]
    assert lines[3].split() == ["cat", "20", "25%", "-1.50"]
    assert lines[4].split() == ["dining", "table", "5", "100%", "+0.12"]
    assert len(lines) == 5


def test_interval_text_shows_the_value_and_its_interval_with_signs():
    assert interval_text(0.1234, -0.05, 1.0) == "+0.123 [-0.050, +1.000]"
