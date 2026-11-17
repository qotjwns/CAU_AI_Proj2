"""
src/mivlm/evaluation/metrics.py 단위 테스트. 기대값은 손으로 계산했다
pytest tests/unit/test_metrics.py -v
"""
import pytest

from mivlm.evaluation.metrics import compare_runs, join_results, summarize, summarize_by


def _row(case_id, label, prediction, margin, variant="PPN"):
    return {"case_id": case_id, "label": label, "prediction": prediction, "logit_margin": margin, "variant": variant}


ROWS = [
    _row("a", "Yes", "Yes", 1.0, "PPP"),    # 맞음
    _row("b", "Yes", "No", -0.5, "PPP"),    # False-No
    _row("c", "No", "Yes", 0.25),           # False-Yes, 경계(|0.25| <= 0.25)
    _row("d", "No", "No", 0.0),             # 맞음, 동점, 경계
    _row("e", "No", "No", -2.0),            # 맞음
    _row("f", "No", None, -1.0),            # 판정 불가 → 오답
]

# ------------------------------------------------------------------- summarize


def test_summarize_hand_computed():
    assert summarize(ROWS) == {
        "n": 6,
        "accuracy": 3 / 6,               # a, d, e
        "false_yes_rate": 1 / 4,         # 정답 No 4개(c, d, e, f) 중 c
        "false_no_rate": 1 / 2,          # 정답 Yes 2개(a, b) 중 b
        "yes_rate": 2 / 6,               # a, c
        "mean_margin": (1.0 - 0.5 + 0.25 + 0.0 - 2.0 - 1.0) / 6,   # -0.375
        "n_tie": 1,                      # d
        "n_boundary": 2,                 # c, d
        "n_none": 1,                     # f
    }


def test_rates_are_none_without_matching_labels():
    only_yes_labels = summarize(ROWS[:2])

    assert only_yes_labels["false_yes_rate"] is None    # 정답이 No인 케이스가 없다
    assert only_yes_labels["false_no_rate"] == 1 / 2


def test_summarize_empty():
    summary = summarize([])

    assert summary["n"] == 0
    assert summary["accuracy"] is None
    assert summary["mean_margin"] is None


# ---------------------------------------------------------------- summarize_by


def test_summarize_by_groups_in_first_seen_order():
    by_variant = summarize_by(ROWS, "variant")

    assert list(by_variant) == ["PPP", "PPN"]
    assert by_variant["PPP"]["n"] == 2
    assert by_variant["PPN"]["n"] == 4
    assert by_variant["PPN"]["false_yes_rate"] == 1 / 4


# ---------------------------------------------------------------- join_results


def test_join_merges_case_and_result_fields():
    cases = [{"case_id": "a", "label": "Yes"}, {"case_id": "b", "label": "No"}]
    results = [{"case_id": "b", "prediction": "No"}, {"case_id": "a", "prediction": "Yes"}]

    assert join_results(cases, results, "case_id") == [
        {"case_id": "a", "label": "Yes", "prediction": "Yes"},
        {"case_id": "b", "label": "No", "prediction": "No"},
    ]


def test_join_rejects_missing_unknown_and_duplicate_results():
    cases = [{"case_id": "a"}, {"case_id": "b"}]

    with pytest.raises(ValueError):
        join_results(cases, [{"case_id": "a"}], "case_id")                                      # b의 결과가 없음
    with pytest.raises(ValueError):
        join_results(cases, [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "x"}], "case_id")  # 모르는 case_id
    with pytest.raises(ValueError):
        join_results(cases, [{"case_id": "a"}, {"case_id": "a"}], "case_id")                    # 중복


# ---------------------------------------------------------------- compare_runs


def test_compare_runs_hand_computed():
    run_a = [_row("a", "No", "No", 0.0), _row("b", "No", "Yes", 0.5), _row("c", "No", "No", -1.0), _row("d", "Yes", "Yes", 2.0)]
    run_b = [_row("a", "No", "Yes", 0.5), _row("b", "No", "No", -0.5), _row("c", "No", "No", -1.0), _row("d", "Yes", "Yes", 1.0)]

    result = compare_runs(run_a, run_b)

    assert result["n"] == 4
    assert result["n_flip"] == 2
    assert result["flips"] == {"No->Yes": 1, "Yes->No": 1}
    assert result["margin_diff_mean"] == (-0.5 + 1.0 + 0.0 + 1.0) / 4    # 0.375
    assert result["margin_abs_diff_max"] == 1.0


def test_compare_runs_rejects_different_cases():
    with pytest.raises(ValueError):
        compare_runs([_row("a", "No", "No", 0.0)], [_row("b", "No", "No", 0.0)])
