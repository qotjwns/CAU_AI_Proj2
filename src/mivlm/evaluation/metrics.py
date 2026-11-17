"""
평가 지표: 추론 결과를 케이스와 합쳐 비율과 margin을 요약한다
"""
import json
import statistics
from collections import Counter
from pathlib import Path

from mivlm.data.cases import load_cases, pilot_cases

BOUNDARY_MARGIN = 0.25  # |margin|이 이 값 이하면 경계 케이스로 센다. bf16 간격(0.125) 두 칸


def join_results(cases: list[dict], results: list[dict]) -> list[dict]:
    """케이스와 결과를 case_id로 합친 행을 만든다 (케이스 순서 유지)

    결과가 빠진 케이스, 케이스 파일에 없는 결과, 중복된 case_id가 있으면 에러를 낸다.
    일부만 있는 결과로 비율을 계산하면 그럴듯하지만 틀린 숫자가 나오기 때문이다.
    """
    results_by_id = {}
    for result in results:
        if result["case_id"] in results_by_id:
            raise ValueError(f"결과에 case_id가 중복됨: {result['case_id']}")
        results_by_id[result["case_id"]] = result
    case_ids = {case["case_id"] for case in cases}
    if set(results_by_id) != case_ids:
        n_missing = len(case_ids - set(results_by_id))
        n_unknown = len(set(results_by_id) - case_ids)
        raise ValueError(f"결과가 없는 케이스 {n_missing}개, 케이스 파일에 없는 결과 {n_unknown}개")
    return [{**case, **results_by_id[case["case_id"]]} for case in cases]


def load_rows(run_dir: Path) -> list[dict]:
    """실행 폴더(meta.json, results.jsonl)의 결과를 케이스와 합친 행으로 읽는다"""
    with open(run_dir / "meta.json") as f:
        pilot = json.load(f)["pilot"]
    with open(run_dir / "results.jsonl") as f:
        results = [json.loads(line) for line in f]
    cases = load_cases()
    return join_results(pilot_cases(cases) if pilot else cases, results)


def _rate(count: int, total: int) -> float | None:
    """count / total. 분모가 0이면 None (0으로 적으면 '해당 오류가 없다'로 잘못 읽힌다)"""
    return count / total if total else None


def summarize(rows: list[dict]) -> dict:
    """한 묶음의 지표

    false_yes_rate: 정답이 No인 케이스 중 Yes로 답한 비율
    false_no_rate: 정답이 Yes인 케이스 중 No로 답한 비율
    yes_rate: 정답과 상관없이 Yes로 답한 비율
    판정 불가(prediction이 None)는 오답으로 세고 n_none에 따로 적는다.
    """
    label_no = [row for row in rows if row["label"] == "No"]
    label_yes = [row for row in rows if row["label"] == "Yes"]
    return {
        "n": len(rows),
        "accuracy": _rate(sum(row["prediction"] == row["label"] for row in rows), len(rows)),
        "false_yes_rate": _rate(sum(row["prediction"] == "Yes" for row in label_no), len(label_no)),
        "false_no_rate": _rate(sum(row["prediction"] == "No" for row in label_yes), len(label_yes)),
        "yes_rate": _rate(sum(row["prediction"] == "Yes" for row in rows), len(rows)),
        "mean_margin": statistics.mean(row["logit_margin"] for row in rows) if rows else None,
        "n_tie": sum(row["logit_margin"] == 0 for row in rows),
        "n_boundary": sum(abs(row["logit_margin"]) <= BOUNDARY_MARGIN for row in rows),
        "n_none": sum(row["prediction"] is None for row in rows),
    }


def summarize_by(rows: list[dict], key: str) -> dict[str, dict]:
    """rows를 key("variant", "object" 등)의 값으로 묶어 묶음마다 summarize한다 (처음 나온 순서 유지)"""
    groups = {}
    for row in rows:
        groups.setdefault(row[key], []).append(row)
    return {name: summarize(group) for name, group in groups.items()}


def compare_runs(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """같은 케이스를 돌린 두 실행을 케이스별로 비교한다

    flips: 답이 바뀐 방향별 개수. 예: {"No->Yes": 32}는 a에서 No, b에서 Yes인 케이스가 32개
    margin_diff: 케이스별 (a의 margin - b의 margin)
    """
    b_by_id = {row["case_id"]: row for row in rows_b}
    if {row["case_id"] for row in rows_a} != set(b_by_id):
        raise ValueError("두 실행의 case_id 집합이 다름")
    flips = Counter()
    diffs = []
    for a in rows_a:
        b = b_by_id[a["case_id"]]
        if a["prediction"] != b["prediction"]:
            flips[f"{a['prediction']}->{b['prediction']}"] += 1
        diffs.append(a["logit_margin"] - b["logit_margin"])
    return {
        "n": len(diffs),
        "n_flip": sum(flips.values()),
        "flips": dict(flips),
        "margin_diff_mean": statistics.mean(diffs),
        "margin_diff_sd": statistics.pstdev(diffs),
        "margin_abs_diff_max": max(abs(diff) for diff in diffs),
    }
