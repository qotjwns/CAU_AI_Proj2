"""
분해 실험: 이미지별 답(B, C)을 케이스 단위의 답으로 합친다

B는 이미지를 1장씩 따로 물은 결과, C는 3장을 넣고 자리를 지목해서 물은 결과다.
케이스의 질문이 "3장 모두에 있는가"이므로 이미지별 답 3개를 AND로 합친다.
요약 표(summarize.py)와 그림(plot.py)이 같은 숫자를 쓰도록, 읽기와 세는 일을 여기서 한다.
"""
from collections import Counter

from mivlm.config import PREDICTIONS_DIR
from mivlm.data.cases import load_cases, load_pools, per_image_queries, single_image_items
from mivlm.evaluation.metrics import ONE_N_VARIANTS, join_results, load_rows, summarize
from mivlm.run_files import read_results


def and_answers(predictions: list[str | None]) -> str | None:
    """이미지별 판정을 AND로 합친다

    하나라도 No면 No (나머지가 무엇이든 '모두에 있다'는 거짓이다).
    전부 Yes면 Yes. No는 없는데 판정 불가(None)가 섞여 있으면 None.
    """
    if "No" in predictions:
        return "No"
    if all(prediction == "Yes" for prediction in predictions):
        return "Yes"
    return None


def _combine(case: dict, image_rows: list[dict]) -> dict:
    """케이스 하나와 그 이미지별 행 3개(자리 순서)를 합친 행. A의 행과 같은 필드를 가진다"""
    return {
        **case,
        "prediction": and_answers([row["prediction"] for row in image_rows]),
        # 세 margin이 모두 0보다 커야 Yes이므로, 가장 작은 margin이 합친 답을 정한다
        "logit_margin": min(row["logit_margin"] for row in image_rows),
        "image_predictions": [row["prediction"] for row in image_rows],
        "image_margins": [row["logit_margin"] for row in image_rows],
    }


def combine_single_image(cases: list[dict], b_rows: list[dict]) -> list[dict]:
    """B: 케이스의 이미지 3장을 이미지 1장 결과에서 찾아 AND로 합친다

    같은 이미지라도 클래스가 다르면 질문이 다르므로 (클래스, 이미지)로 찾는다.
    """
    by_image = {(row["category_id"], row["image_id"]): row for row in b_rows}
    return [
        _combine(case, [by_image[(case["category_id"], image_id)] for image_id in case["image_ids"]])
        for case in cases
    ]


def combine_per_image(cases: list[dict], c_rows: list[dict]) -> list[dict]:
    """C: 케이스의 자리 1, 2, 3에 대한 답을 AND로 합친다"""
    by_slot = {(row["case_id"], row["slot"]): row for row in c_rows}
    return [_combine(case, [by_slot[(case["case_id"], slot)] for slot in (1, 2, 3)]) for case in cases]


def images_all_correct(row: dict) -> bool:
    """합친 행의 이미지별 답 3개가 모두 맞았는지

    변형 이름의 글자가 자리별 정답이다: P 자리는 Yes, N 자리는 No가 맞다.
    AND로 합친 답이 맞아도 이 값은 False일 수 있다.
    예: PPN에서 [No, Yes, No]는 AND가 No로 정답이지만 자리 1의 P를 놓쳤다.
    """
    expected = ["Yes" if letter == "P" else "No" for letter in row["variant"]]
    return row["image_predictions"] == expected


def load_case_rows(run: str) -> tuple[dict[str, list[dict]], list[dict], list[dict]]:
    """실행 이름 하나의 A, B, C 결과를 읽어 케이스 단위의 행으로 맞춘다 (전체 실행 결과용)

    run: 실행 폴더 이름. 예: "qwen2.5-vl-3b"
        A는 01_baseline/<run>/에서, B와 C는 02_decomposition/ 아래 같은 이름의 폴더에서 읽는다.
    반환: (case_rows, b_image_rows, c_image_rows)
        case_rows: {"A": 행들, "B": 행들, "C": 행들}. 셋 다 케이스 2,400개가 같은 순서로 들어 있고,
            B와 C는 이미지별 답을 AND로 합친 것이다
        b_image_rows: B의 이미지별 답 (이미지 1장 항목 320개와 합친 행)
        c_image_rows: C의 이미지별 답 (자리별 질문 항목 7,200개와 합친 행)
    """
    cases = load_cases()
    b_image_rows = join_results(single_image_items(load_pools()),
                                read_results(PREDICTIONS_DIR / "02_decomposition" / "b_single_image" / run), "item_id")
    c_image_rows = join_results(per_image_queries(cases),
                                read_results(PREDICTIONS_DIR / "02_decomposition" / "c_per_image_answers" / run), "query_id")
    case_rows = {
        "A": load_rows(PREDICTIONS_DIR / "01_baseline" / run),
        "B": combine_single_image(cases, b_image_rows),
        "C": combine_per_image(cases, c_image_rows),
    }
    return case_rows, b_image_rows, c_image_rows


def main_metrics(rows: list[dict]) -> dict:
    """메인 결과의 지표 셋: 전체 정확도, N이 1장인 변형의 False-Yes rate, PPP 정확도"""
    return {
        "accuracy": summarize(rows)["accuracy"],
        "one_n_false_yes_rate": summarize([row for row in rows if row["variant"] in ONE_N_VARIANTS])["false_yes_rate"],
        "ppp_accuracy": summarize([row for row in rows if row["variant"] == "PPP"])["accuracy"],
    }


def a_false_yes_answers(case_rows: dict[str, list[dict]]) -> Counter:
    """A가 False-Yes인 케이스를 (B의 답, C의 답)으로 나눠 센다. 개수의 합이 A의 False-Yes 케이스 수다

    case_rows: load_case_rows의 첫 번째 결과. B와 C의 답은 AND로 합친 케이스 단위의 답이다
    예: {("No", "No"): 7, ("No", "Yes"): 3}은 B, C 모두 맞힌 케이스가 7개, B만 맞힌 케이스가 3개
    """
    b_answers = {row["case_id"]: row["prediction"] for row in case_rows["B"]}
    c_answers = {row["case_id"]: row["prediction"] for row in case_rows["C"]}
    return Counter((b_answers[row["case_id"]], c_answers[row["case_id"]])
                   for row in case_rows["A"] if row["label"] == "No" and row["prediction"] == "Yes")


def one_n_outcomes(rows: list[dict]) -> dict:
    """N이 1장인 케이스(정답 No)에서 AND로 합친 답이 가리는 이미지별 오류를 센다

    rows: combine_single_image 또는 combine_per_image의 결과
    n: N이 1장인 케이스 수
    answered_no: 그중 합친 답이 No인(맞힌) 케이스 수
    all_images_correct: 그중 이미지별 답 3개도 모두 맞은 케이스 수. answered_no와의 차이가 AND에 가려진 오류다
    false_yes: 합친 답이 Yes인 케이스 수. 판정 불가(None)가 있으면 answered_no + false_yes가 n보다 작다
    """
    one_n = [row for row in rows if row["variant"] in ONE_N_VARIANTS]
    answered_no = [row for row in one_n if row["prediction"] == "No"]
    return {
        "n": len(one_n),
        "answered_no": len(answered_no),
        "all_images_correct": sum(images_all_correct(row) for row in answered_no),
        "false_yes": sum(row["prediction"] == "Yes" for row in one_n),
    }
