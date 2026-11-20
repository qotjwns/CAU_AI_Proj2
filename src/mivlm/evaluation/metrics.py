"""
평가 지표: 추론 결과를 케이스와 합쳐 비율과 margin을 요약한다
"""
import statistics
from collections import Counter
from pathlib import Path

from mivlm.data.cases import load_cases, pilot_cases, run_pool
from mivlm.run_files import read_meta, read_results

BOUNDARY_MARGIN = 0.25  # |margin|이 이 값 이하면 경계 케이스로 센다. bf16 간격(0.125) 두 칸
VARIANT_ORDER = ["PPP", "PPN", "PNP", "NPP", "PNN", "NNN"]  # 표와 그림에서 변형을 늘어놓는 순서: N 개수 순 (0, 1, 1, 1, 2, 3)
ONE_N_VARIANTS = ["PPN", "PNP", "NPP"]  # N이 1장인 변형. 정답은 No이고, False-Yes가 주로 나오는 변형이다


def join_results(items: list[dict], results: list[dict], id_key: str) -> list[dict]:
    """항목(케이스 등)과 결과를 id_key 필드로 합친 행을 만든다 (항목 순서 유지)

    id_key: 항목을 구분하는 필드 이름. A는 "case_id", B는 "item_id", C는 "query_id"
    결과가 빠진 항목, 항목 목록에 없는 결과, 중복된 id가 있으면 에러를 낸다.
    일부만 있는 결과로 비율을 계산하면 그럴듯하지만 틀린 숫자가 나오기 때문이다.
    """
    results_by_id = {}
    for result in results:
        if result[id_key] in results_by_id:
            raise ValueError(f"결과에 {id_key}가 중복됨: {result[id_key]}")
        results_by_id[result[id_key]] = result
    item_ids = {item[id_key] for item in items}
    if set(results_by_id) != item_ids:
        n_missing = len(item_ids - set(results_by_id))
        n_unknown = len(set(results_by_id) - item_ids)
        raise ValueError(f"결과가 없는 항목 {n_missing}개, 항목 목록에 없는 결과 {n_unknown}개")
    return [{**item, **results_by_id[item[id_key]]} for item in items]


def load_rows(run_dir: Path) -> list[dict]:
    """A(01_baseline) 실행 폴더의 결과를 케이스와 합친 행으로 읽는다. 파일럿 실행이면 파일럿 케이스와 합친다

    케이스는 실행 폴더의 meta.json이 가리키는 풀에서 읽고, 실행 당시의 데이터 해시와 같은지 확인한다 (run_pool).
    파일럿 케이스도 그 풀의 규칙으로 고른다 (pilot_cases).
    """
    pool = run_pool(run_dir)
    cases = load_cases(pool)
    if read_meta(run_dir)["pilot"]:
        cases = pilot_cases(cases, pool)
    return join_results(cases, read_results(run_dir), "case_id")


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
