"""
추론 출력의 형식 점검: 실제 첫 생성 토큰, 파싱, 파싱한 답과 margin 부호의 일치

모델을 바꿨을 때(예: InternVL) 출력의 뜻이 Qwen 실험과 같은지 보는 데 쓴다. 정확도는 보지 않는다.
margin은 가공 전 logit에서, 생성된 답은 생성 후처리(예: repetition penalty)를 거친 점수에서 나온다.
그래서 두 값이 어긋나는 결과는 margin 계산이 틀렸다는 뜻이 아니라 따로 살펴볼 대상이다. 어긋난 결과의 id를 남긴다.
"""
import statistics
from collections import Counter

ANSWER_TOKENS = ("Yes", "No")  # 첫 생성 토큰이 디코딩했을 때 정확히 이 문자열이어야 margin이 답의 첫 토큰과 대응한다
ID_KEYS = ("case_id", "item_id", "query_id")  # A, B, C 결과의 id 필드


def _id(result: dict) -> str:
    return next(result[key] for key in ID_KEYS if key in result)


def _spread(values: list[int]) -> dict[str, float]:
    return {"min": min(values), "mean": statistics.mean(values), "max": max(values)} if values else {}


def output_checks(results: list[dict]) -> dict:
    """results.jsonl의 결과들을 점검한 값

    n                                  결과 수
    first_token_recorded               first_token이 기록된 결과 수 (예전 실행에는 없다)
    first_token_counts                 첫 토큰 문자열별 개수 (많은 순, repr로 적어 공백과 특수 토큰이 보이게)
    first_token_not_answer             첫 토큰이 정확히 "Yes"나 "No"가 아닌 결과의 id
    first_token_vs_prediction_mismatch 첫 토큰이 Yes/No인데 파싱한 답과 다른 결과의 id
    unparsed                           파싱하지 못한(prediction이 None인) 결과의 id
    ties                               margin이 정확히 0인 결과 수
    sign_mismatch                      파싱한 답과 margin 부호가 다른 결과의 id (파싱 실패와 margin 0은 뺀다)
    input_tokens, image_tokens         입력 토큰 수와 이미지 토큰 수의 최솟값, 평균, 최댓값
    """
    recorded = [result for result in results if "first_token" in result]
    return {
        "n": len(results),
        "first_token_recorded": len(recorded),
        "first_token_counts": {repr(token): count
                               for token, count in Counter(result["first_token"] for result in recorded).most_common()},
        "first_token_not_answer": [_id(result) for result in recorded if result["first_token"] not in ANSWER_TOKENS],
        "first_token_vs_prediction_mismatch": [
            _id(result) for result in recorded
            if result["first_token"] in ANSWER_TOKENS and result["prediction"] != result["first_token"]],
        "unparsed": [_id(result) for result in results if result["prediction"] is None],
        "ties": sum(result["logit_margin"] == 0 for result in results),
        "sign_mismatch": [_id(result) for result in results
                          if result["prediction"] is not None and result["logit_margin"] != 0
                          and (result["prediction"] == "Yes") != (result["logit_margin"] > 0)],
        "input_tokens": _spread([result["input_token_count"] for result in results]),
        "image_tokens": _spread([result["image_token_count"] for result in results]),
    }
