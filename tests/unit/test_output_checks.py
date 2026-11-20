"""
src/mivlm/evaluation/output_checks.py 단위 테스트 (가짜 결과만 쓴다)
pytest tests/unit/test_output_checks.py -v
"""
from mivlm.evaluation.output_checks import output_checks


def _result(case_id: str, first: str | None, prediction: str | None, margin: float) -> dict:
    result = {"case_id": case_id, "prediction": prediction, "logit_margin": margin,
              "input_token_count": 100, "image_token_count": 80}
    if first is not None:
        result["first_token"] = first
    return result


def test_clean_outputs_have_no_flagged_results():
    checks = output_checks([_result("a", "Yes", "Yes", 1.5), _result("b", "No", "No", -0.5)])

    assert checks["n"] == checks["first_token_recorded"] == 2
    assert checks["first_token_counts"] == {"'Yes'": 1, "'No'": 1}
    for key in ["first_token_not_answer", "first_token_vs_prediction_mismatch", "unparsed", "sign_mismatch"]:
        assert checks[key] == []
    assert checks["ties"] == 0
    assert checks["input_tokens"] == {"min": 100, "mean": 100, "max": 100}


def test_each_kind_of_problem_is_listed_by_id():
    checks = output_checks([
        _result("think", "<think>", None, 2.0),        # 첫 토큰이 답이 아니고 파싱도 실패
        _result("space", " Yes", "Yes", 1.0),          # 앞 공백이 붙은 토큰은 "Yes"가 아니다
        _result("flip", "Yes", "Yes", -0.25),          # 답은 Yes인데 가공 전 margin은 No 쪽
        _result("tie", "No", "No", 0.0),               # margin 0은 부호 비교에서 뺀다
        _result("text", "No", "Yes", 0.5),             # 첫 토큰과 파싱한 답이 다르다
    ])

    assert checks["first_token_not_answer"] == ["think", "space"]
    assert checks["unparsed"] == ["think"]
    assert checks["sign_mismatch"] == ["flip"]
    assert checks["first_token_vs_prediction_mismatch"] == ["text"]
    assert checks["ties"] == 1
    assert checks["first_token_counts"]["'<think>'"] == 1 and checks["first_token_counts"]["' Yes'"] == 1


def test_old_results_without_first_token_are_counted_and_ids_follow_the_result_kind():
    results = [{"item_id": "cat_1", "prediction": "No", "logit_margin": -1.0,
                "input_token_count": 10, "image_token_count": 5}]

    checks = output_checks(results)

    assert checks["first_token_recorded"] == 0 and checks["first_token_not_answer"] == []
    assert checks["sign_mismatch"] == [] and checks["n"] == 1
