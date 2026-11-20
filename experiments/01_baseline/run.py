"""
STEP 1 baseline (A): 이미지 3장을 한꺼번에 넣고 최종 Yes/No를 받는다
결과는 outputs/predictions/01_baseline/<모델>[_float32][_pilot]/에 저장된다
python experiments/01_baseline/run.py --pilot     # 클래스별 1개, 20케이스 (Mac)
python experiments/01_baseline/run.py             # 전체 2,400케이스 (GPU 서버)
python experiments/01_baseline/run.py --dtype float32     # float32로 실행, bf16 결과와 비교용 (GPU 서버)

중간에 끊기면 같은 명령으로 다시 실행한다. 이미 끝난 케이스는 건너뛴다.
"""
from mivlm.data.cases import load_cases, pilot_cases
from mivlm.models.inference import ALL_IMAGES_QUESTION, case_messages
from mivlm.runner import parse_run_args, run_inference


def main() -> None:
    args = parse_run_args("STEP 1 baseline 추론")
    cases = load_cases(args.pool)
    if args.pilot:
        cases = pilot_cases(cases, args.pool)
    run_inference("01_baseline", args, cases, "case_id", case_messages, ALL_IMAGES_QUESTION)


if __name__ == "__main__":
    main()
