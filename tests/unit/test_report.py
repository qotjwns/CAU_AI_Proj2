"""
src/mivlm/evaluation/report.py의 층별 표 출력 단위 테스트
pytest tests/unit/test_report.py -v
"""
import torch

from mivlm.evaluation.report import print_layer_table


def test_layer_table_has_one_row_per_layer_and_one_column_per_name(capsys):
    print_layer_table("제목", {1: torch.tensor([1.0, 0.5]), "2*": torch.tensor([0.25, 0.125])}, ".3f")

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ""
    assert lines[1] == "[제목]"
    assert lines[2].split() == ["layer", "1", "2*"]
    assert lines[3].split() == ["0", "1.000", "0.250"]
    assert lines[4].split() == ["1", "0.500", "0.125"]
    assert len(lines) == 5
