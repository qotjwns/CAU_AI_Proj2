"""
분해 실험 요약: 같은 2,400개 케이스에 대한 A(3장 동시), B(1장씩 AND), C(자리별 질문 AND)를 비교한다
python experiments/02_decomposition/summarize.py qwen2.5-vl-3b

A는 outputs/predictions/01_baseline/<실행 이름>/에서, B와 C는 02_decomposition/ 아래 같은 이름의 폴더에서 읽는다 (전체 실행 결과용).
케이스별 원본 표는 outputs/tables/02_decomposition/<실행 이름>_cases.csv로 저장한다.
"""
import argparse
import csv
from collections import Counter

from mivlm.config import PREDICTIONS_DIR, PROJECT_ROOT
from mivlm.data.cases import load_cases
from mivlm.evaluation.decomposition import (
    combine_per_image, combine_single_image, images_all_correct, load_per_image_rows, load_single_image_rows,
)
from mivlm.evaluation.metrics import VARIANT_ORDER, load_rows, summarize, summarize_by
from mivlm.evaluation.report import format_value, print_table

ONE_N_VARIANTS = ["PPN", "PNP", "NPP"]  # N이 1장인 변형


def percent(count: int, total: int) -> str:
    return f"{count / total:.1%}" if total else "-"


def print_columns(title: str, row_names: list[str], columns: dict[str, list[str]]) -> None:
    """행 이름과 A, B, C 열을 나란히 출력한다. columns: {열 이름: 행 순서대로의 값}"""
    print(f"\n[{title}]")
    print(f"{'':<34}" + "".join(f"{name:>16}" for name in columns))
    for i, row_name in enumerate(row_names):
        print(f"{row_name:<34}" + "".join(f"{values[i]:>16}" for values in columns.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description="분해 실험 A, B, C 비교")
    parser.add_argument("run", help="실행 폴더 이름. 예: qwen2.5-vl-3b")
    args = parser.parse_args()

    # 이미지별 답 (B: 320개, C: 7,200개)
    b_image_rows = load_single_image_rows(PREDICTIONS_DIR / "02_decomposition" / "b_single_image" / args.run)
    c_image_rows = load_per_image_rows(PREDICTIONS_DIR / "02_decomposition" / "c_per_image_answers" / args.run)
    # 케이스 단위의 답 (각 2,400개). B와 C는 이미지별 답을 AND로 합친 것
    cases = load_cases()
    case_rows = {
        "A (3장 동시)": load_rows(PREDICTIONS_DIR / "01_baseline" / args.run),
        "B (1장씩 AND)": combine_single_image(cases, b_image_rows),
        "C (자리별 AND)": combine_per_image(cases, c_image_rows),
    }
    a_rows, b_rows, c_rows = case_rows.values()

    # 1. 메인 결과표
    # 형식 오류율: 모델의 답이 Yes도 No도 아닌 비율. A는 케이스 2,400개, B는 답 320개, C는 답 7,200개 기준
    answer_rows = {"A (3장 동시)": a_rows, "B (1장씩 AND)": b_image_rows, "C (자리별 AND)": c_image_rows}
    main_table = {}
    for name, rows in case_rows.items():
        one_n = summarize([row for row in rows if row["variant"] in ONE_N_VARIANTS])
        ppp = summarize([row for row in rows if row["variant"] == "PPP"])
        unparsed = sum(row["prediction"] is None for row in answer_rows[name])
        main_table[name] = [
            format_value("accuracy", summarize(rows)["accuracy"]),
            format_value("false_yes_rate", one_n["false_yes_rate"]),
            format_value("accuracy", ppp["accuracy"]),
            percent(unparsed, len(answer_rows[name])),
        ]
    print_columns("1. 메인 결과", ["전체 정확도", "N 1장(PPN, PNP, NPP) False-Yes", "PPP 정확도", "형식 오류율"], main_table)

    # 2. 변형별 Yes 응답 비율. PPP는 Yes가 정답이고, 나머지는 이 비율이 False-Yes rate다
    by_variant = {name: summarize_by(rows, "variant") for name, rows in case_rows.items()}
    print_columns("2. 변형별 Yes로 답한 비율 (PPP만 Yes가 정답)", VARIANT_ORDER,
                  {name: [format_value("yes_rate", by_variant[name][variant]["yes_rate"]) for variant in VARIANT_ORDER]
                   for name in case_rows})

    # 3. 케이스별 원본 표: 파일로 저장
    table_path = PROJECT_ROOT / "outputs" / "tables" / "02_decomposition" / f"{args.run}_cases.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with open(table_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "object", "variant", "label", "a", "a_margin",
                         "b_images", "b_and", "b_min_margin", "c_images", "c_and", "c_min_margin"])
        for a, b, c in zip(a_rows, b_rows, c_rows):
            writer.writerow([a["case_id"], a["object"], a["variant"], a["label"], a["prediction"], a["logit_margin"],
                             ",".join(str(p) for p in b["image_predictions"]), b["prediction"], b["logit_margin"],
                             ",".join(str(p) for p in c["image_predictions"]), c["prediction"], c["logit_margin"]])
    print(f"\n[3. 케이스별 원본 표] {len(a_rows)}줄 -> {table_path}")

    # 4. A가 False-Yes인 케이스를 B, C의 결과로 나눈다
    b_by_id = {row["case_id"]: row for row in b_rows}
    c_by_id = {row["case_id"]: row for row in c_rows}
    a_false_yes = [row for row in a_rows if row["label"] == "No" and row["prediction"] == "Yes"]
    counts = Counter((b_by_id[row["case_id"]]["prediction"], c_by_id[row["case_id"]]["prediction"]) for row in a_false_yes)
    print(f"\n[4. A가 False-Yes인 케이스 {len(a_false_yes)}개: B와 C의 답]")
    meanings = {("No", "No"): "B, C 모두 맞힘", ("No", "Yes"): "B만 맞힘", ("Yes", "No"): "C만 맞힘", ("Yes", "Yes"): "B, C 모두 False-Yes"}
    for (b_answer, c_answer), n in sorted(counts.items(), key=lambda item: -item[1]):
        meaning = meanings.get((b_answer, c_answer), "판정 불가 포함")
        print(f"  B {str(b_answer):<5} C {str(c_answer):<5} {n:>5}  {percent(n, len(a_false_yes)):>7}  {meaning}")

    # 5. 이미지별 판단. AND는 No가 하나만 있어도 No가 되므로, 케이스 정확도만으로는 이미지별 오류가 가려진다
    #    false_no_rate: P 이미지를 No라고 한 비율 / false_yes_rate: N 이미지를 Yes라고 한 비율
    image_tables = {"B (1장만 넣음)": summarize(b_image_rows), "C (3장 중 지목)": summarize(c_image_rows)}
    for slot in (1, 2, 3):
        image_tables[f"C 자리 {slot}"] = summarize([row for row in c_image_rows if row["slot"] == slot])
    print_table("5. 이미지별 판단", image_tables)

    # 6. AND가 가리는 오류: 케이스의 답은 맞았지만 이미지별 답은 틀린 경우를 드러낸다
    #    N이 1장인 케이스는 N 자리만 No라고 하면 AND가 No로 정답이 되므로, P 자리를 No라고 틀려도 가려진다
    hidden_table = {}
    for name, rows in [("B (1장씩 AND)", b_rows), ("C (자리별 AND)", c_rows)]:
        one_n = [row for row in rows if row["variant"] in ONE_N_VARIANTS]
        one_n_correct = [row for row in one_n if row["prediction"] == "No"]
        hidden_table[name] = [
            percent(sum(images_all_correct(row) for row in rows), len(rows)),
            f"{len(one_n_correct)} / {len(one_n)}",
            percent(sum(images_all_correct(row) for row in one_n_correct), len(one_n_correct)),
        ]
    print_columns("6. AND가 가리는 오류",
                  ["세 이미지 답이 모두 맞은 케이스", "N 1장 케이스 중 No로 맞힌 수", "  그중 세 이미지 답이 모두 맞음"], hidden_table)

    # 7. B: 클래스별 이미지 1장 판단 (클래스마다 P 8장, N 8장)
    print_table("7. B의 클래스별 이미지 1장 판단", summarize_by(b_image_rows, "object"))


if __name__ == "__main__":
    main()
