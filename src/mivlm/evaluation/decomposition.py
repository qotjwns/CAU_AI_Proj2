"""
분해 실험: 이미지별 답(B, C)을 케이스 단위의 답으로 합친다

B는 이미지를 1장씩 따로 물은 결과, C는 3장을 넣고 자리를 지목해서 물은 결과다.
케이스의 질문이 "3장 모두에 있는가"이므로 이미지별 답 3개를 AND로 합친다.
"""
from pathlib import Path

from mivlm.data.cases import load_cases, load_pools, per_image_queries, single_image_items
from mivlm.evaluation.metrics import join_results, read_results


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


def load_single_image_rows(run_dir: Path) -> list[dict]:
    """B 실행 폴더의 결과를 이미지 1장 항목(320개)과 합친 행으로 읽는다. 전체 실행 결과용"""
    return join_results(single_image_items(load_pools()), read_results(run_dir), "item_id")


def load_per_image_rows(run_dir: Path) -> list[dict]:
    """C 실행 폴더의 결과를 자리별 질문 항목(7,200개)과 합친 행으로 읽는다. 전체 실행 결과용"""
    return join_results(per_image_queries(load_cases()), read_results(run_dir), "query_id")
