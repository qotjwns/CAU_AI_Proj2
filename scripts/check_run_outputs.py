"""
실행 폴더의 추론 출력 형식을 점검해 출력한다 (모델은 쓰지 않는다. 정의는 mivlm.evaluation.output_checks)
python scripts/check_run_outputs.py 01_baseline/internvl3.5-4b_dev_pilot
python scripts/check_run_outputs.py 02_decomposition/b_single_image/internvl3.5-4b_dev_pilot

파일럿 통과 기준: 첫 토큰이 Yes/No가 아닌 결과, 첫 토큰과 파싱한 답이 다른 결과, 파싱 실패, 답과 margin 부호의 불일치가
모두 0개. 하나라도 있으면 다음 단계로 넘어가지 않고 그 결과들의 원인을 먼저 본다.
"""
import argparse
import json

from mivlm.config import PREDICTIONS_DIR
from mivlm.data.cases import run_pool
from mivlm.evaluation.output_checks import output_checks
from mivlm.run_files import read_meta, read_results

LISTS = ["first_token_not_answer", "first_token_vs_prediction_mismatch", "unparsed", "sign_mismatch"]


def main() -> None:
    parser = argparse.ArgumentParser(description="추론 출력의 형식 점검")
    parser.add_argument("run", help="outputs/predictions/ 아래의 실행 폴더 경로 (예: 01_baseline/internvl3.5-4b_dev_pilot)")
    args = parser.parse_args()

    run_dir = PREDICTIONS_DIR / args.run
    meta = read_meta(run_dir)
    pool = run_pool(run_dir)  # 데이터 출처도 같이 확인한다
    checks = output_checks(read_results(run_dir))

    print(f"[실행] {args.run}: 모델 {meta['model']['repo_id']} @ {meta['model']['commit']}, {meta['model']['dtype']}, "
          f"{meta['model']['device']}, 풀 {pool or '기존 탐색 데이터'}, 파일럿 {meta['pilot']}")
    if "loading" in meta:
        loading = meta["loading"]
        print(f"[로딩] {loading['model_class']}, {loading['processor_class']} @ {loading['processor_commit']}, "
              f"chat template {loading['chat_template_sha256']}, tokenizers {loading['tokenizers']}")
    print(f"[생성] {meta['generation']}")
    print(f"결과 {checks['n']}개, 첫 토큰 기록 {checks['first_token_recorded']}개, margin 0 {checks['ties']}개")
    print(f"첫 토큰 분포: {checks['first_token_counts']}")
    print(f"입력 토큰 {checks['input_tokens']}, 이미지 토큰 {checks['image_tokens']}")
    for key in LISTS:
        print(f"{key}: {len(checks[key])}개 {json.dumps(checks[key][:20], ensure_ascii=False)}")
    passed = all(not checks[key] for key in LISTS) and checks["first_token_recorded"] == checks["n"]
    print(f"\n[판정] {'통과' if passed else '확인 필요'}")


if __name__ == "__main__":
    main()
