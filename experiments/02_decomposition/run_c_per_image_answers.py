"""
분해 실험 C: 이미지 3장을 한꺼번에 넣고 자리 하나씩 지목 (케이스 2,400개 x 자리 3개 = 7,200번)
결과는 outputs/predictions/02_decomposition/c_per_image_answers/<모델>[_float32][_pilot]/에 저장된다
python experiments/02_decomposition/run_c_per_image_answers.py --pilot     # 20케이스 x 3, 60개 (Mac)
python experiments/02_decomposition/run_c_per_image_answers.py             # 전체 7,200개 (GPU 서버)

B와 다른 점은 다른 두 이미지가 같이 들어 있다는 것뿐이다. 케이스별 답(AND 결합)은 summarize.py에서 만든다.
"""
from mivlm.data.cases import load_cases, per_image_queries, pilot_cases
from mivlm.models.inference import PER_IMAGE_QUESTION, per_image_messages
from mivlm.runner import parse_run_args, run_inference


def main() -> None:
    args = parse_run_args("분해 실험 C: 이미지 3장을 넣고 자리별로 묻기")
    cases = load_cases(args.pool)
    if args.pilot:
        cases = pilot_cases(cases, args.pool)
    run_inference("02_decomposition/c_per_image_answers", args, per_image_queries(cases), "query_id",
                  per_image_messages, PER_IMAGE_QUESTION)


if __name__ == "__main__":
    main()
