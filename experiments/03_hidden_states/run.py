"""
STEP 5 hidden state 저장: pair마다 변형 6개를 forward 해서 케이스별 값과 12개 비교를 저장한다 (답은 생성하지 않는다)
결과는 outputs/predictions/03_hidden_states/<모델>[_float32][_pilot]/에 저장된다
    meta.json         실행 조건
    results.jsonl     pair마다 한 줄: 파일 이름과 해시, 케이스별 logit, 위치 통제 여부, same_prefix 기준선
    pairs/<pair>.pt   pair마다 파일 하나: 케이스별 pre_answer_hidden, 비교별 토큰 유사도
                      (내용은 mivlm.analysis.pair_record의 case_record, comparison_record 참고)
python experiments/03_hidden_states/run.py --pilot     # pair 3개, forward 18번 (Mac)
python experiments/03_hidden_states/run.py             # 전체 400 pair, forward 2,400번 (GPU 서버)

중간에 끊기면 같은 명령으로 다시 실행한다. 이미 끝난 pair는 건너뛴다.
실행이 끝나면 same_prefix 기준선의 측정값과, baseline(01_baseline) 결과와의 logit margin 비교를 출력한다.
"""
from pathlib import Path

from mivlm.analysis.pair_record import build_pair_record, observe_case, pair_summary
from mivlm.analysis.pair_store import check_saved_pairs, save_pair_record
from mivlm.config import PREDICTIONS_DIR
from mivlm.data.cases import case_pairs, load_cases
from mivlm.evaluation.metrics import compare_runs
from mivlm.models.inference import ALL_IMAGES_QUESTION
from mivlm.run_files import read_meta, read_results
from mivlm.runner import parse_run_args, run_items, start_run

EXPERIMENT = "03_hidden_states"
# 파일럿에 쓰는 pair 3개 (Mac에서 값을 미리 확인해 둔 pair)
PILOT_PAIRS = ["group_05_cat_4", "group_02_bed_2", "group_03_bowl_2"]


def run_pair(model, processor, run_dir: Path, item: dict) -> dict:
    """pair 하나: 변형 6개를 forward → 기록 만들기 → 파일 저장. results.jsonl에 적을 요약을 돌려준다

    item: {"pair": pair 이름, "cases": {변형: 케이스}}
    observation 6개(각각 수백 MB)는 이 함수 안에서만 쓰이고, 함수가 끝나면 메모리에서 사라진다.
    """
    observations = {variant: observe_case(model, processor, case) for variant, case in item["cases"].items()}
    record = build_pair_record(item["pair"], observations)
    return {**save_pair_record(run_dir, record), **pair_summary(record)}


def report_same_prefix(results: list[dict]) -> None:
    """same_prefix 기준선의 측정값을 출력한다: 앞부분이 같은 두 입력에서 벡터가 완전히 같았는가

    완전히 같지 않은 pair는 차이가 큰 순서로 10개까지 적는다.
    """
    different = sorted((result for result in results if not result["same_prefix_identical"]),
                       key=lambda result: -result["same_prefix_max_relative_l2"])
    largest = max(result["same_prefix_max_relative_l2"] for result in results)
    print(f"\n[same_prefix 기준선] pair {len(results)}개: 완전히 같음 {len(results) - len(different)}개, "
          f"다름 {len(different)}개, Relative L2 최댓값 {largest:.3g}")
    for result in different[:10]:
        print(f"  {result['pair']}: Relative L2 최댓값 {result['same_prefix_max_relative_l2']:.3g}")


def report_baseline_margins(run_dir: Path, results: list[dict]) -> None:
    """같은 케이스의 baseline 결과와 logit margin을 비교해 출력한다. baseline 결과가 없으면 건너뛴다

    비교 대상은 같은 모델과 dtype으로 돌린 01_baseline의 전체 실행이다.
    이 실행은 답을 생성하지 않으므로 답을 margin의 부호로 본다 (양수면 Yes).
    기록된 조건(기기, dtype, 패키지 버전)이 같아도 margin이 완전히 같지는 않을 수 있다. 값이 촘촘한 float32에서는
    반올림 수준의 차이가 그대로 드러난다 (baseline은 generate, 이 실행은 forward이고, GPU 기기도 다를 수 있다).
    그래서 두 실행의 조건, 완전히 같은 케이스 수, 차이의 최댓값을 함께 적고, 판정은 하지 않는다.
    """
    baseline_dir = PREDICTIONS_DIR / "01_baseline" / run_dir.name.removesuffix("_pilot")
    if not (baseline_dir / "results.jsonl").exists():
        print(f"\n[baseline 비교] {baseline_dir}에 결과가 없어서 건너뜀")
        return

    print(f"\n[baseline 비교] {baseline_dir}")
    for name, directory in [("이번 실행", run_dir), ("baseline", baseline_dir)]:
        meta = read_meta(directory)
        print(f"  {name}: {meta['model']['device']}, {meta['model']['dtype']}, torch {meta['environment']['torch']}, "
              f"transformers {meta['environment']['transformers']}, torchcodec {meta['preprocessing']['torchcodec']}")

    rows = [{"case_id": case["case_id"], "logit_margin": case["logit_margin"],
             "prediction": "Yes" if case["logit_margin"] > 0 else "No"}
            for result in results for case in result["cases"].values()]
    case_ids = {row["case_id"] for row in rows}
    baseline_by_id = {row["case_id"]: row for row in read_results(baseline_dir) if row["case_id"] in case_ids}

    comparison = compare_runs(rows, list(baseline_by_id.values()))
    n_exact = sum(row["logit_margin"] == baseline_by_id[row["case_id"]]["logit_margin"] for row in rows)
    print(f"  케이스 {comparison['n']}개: margin이 완전히 같음 {n_exact}개, "
          f"차이의 최댓값 {comparison['margin_abs_diff_max']:.3g}, 답이 다름 {comparison['n_flip']}개 {comparison['flips']}")
    for row in rows:  # 답이 달라진 케이스. margin이 0 근처인지 볼 수 있게 두 값을 다 적는다
        baseline = baseline_by_id[row["case_id"]]
        if row["prediction"] != baseline["prediction"]:
            print(f"    {row['case_id']}: 이번 {row['logit_margin']:+.3f}, "
                  f"baseline {baseline['logit_margin']:+.3f} ({baseline['prediction']})")


def main() -> None:
    args = parse_run_args("STEP 5 hidden state 저장")
    pairs = case_pairs(load_cases(args.pool))
    if args.pilot:
        pairs = [item for item in pairs if item["pair"] in PILOT_PAIRS]
        if len(pairs) != len(PILOT_PAIRS):
            raise ValueError(f"파일럿 pair {PILOT_PAIRS} 중 cases.jsonl에 없는 것이 있음")

    run_dir, model, processor = start_run(EXPERIMENT, args, len(pairs), ALL_IMAGES_QUESTION)
    check_saved_pairs(run_dir)
    run_items(run_dir, pairs, "pair", lambda item: run_pair(model, processor, run_dir, item))

    results = read_results(run_dir)
    report_same_prefix(results)
    report_baseline_margins(run_dir, results)


if __name__ == "__main__":
    main()
