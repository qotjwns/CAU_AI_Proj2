"""
hidden state 실행의 사진 비교 요약: 같은 사진의 표현이 자리와 앞 사진에 따라 얼마나 달라지는지를 층별 표로 출력
python experiments/03_hidden_states/summarize.py qwen2.5-vl-3b_float32

값은 pair들의 중앙값이다. 결과는 터미널에만 출력한다.
사진 비교는 same_prefix 기준선(달라진 것이 없는 비교)이 0에 가까운 실행에서만 믿을 수 있다.
기준선이 크면 계산 오차가 섞인 것이므로 경고를 출력한다 (CUDA의 bf16 실행이 그렇다. 그때는 float32 실행을 쓴다).
"""
import argparse
import json

from mivlm.analysis.comparisons import (
    ONE_IMAGE_CHANGED, is_position_controlled, load_comparison_rows, median_by_layer, same_prefix_baseline,
)
from mivlm.analysis.hidden_states import SAME_WITHIN
from mivlm.analysis.pair_record import COMPARISONS
from mivlm.config import PREDICTIONS_DIR
from mivlm.evaluation.report import print_layer_table

EXPERIMENT = "03_hidden_states"


def main() -> None:
    parser = argparse.ArgumentParser(description="hidden state 실행의 사진 비교 요약")
    parser.add_argument("run", help="outputs/predictions/03_hidden_states/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    run_dir = PREDICTIONS_DIR / EXPERIMENT / args.run
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    rows = load_comparison_rows(run_dir)
    by_index = {index: [row for row in rows if row["index"] == index] for index in range(len(COMPARISONS))}
    n_pairs = len(by_index[0])
    print(f"{args.run}: pair {n_pairs}개, {meta['model']['name']}, {meta['model']['device']}, {meta['model']['dtype']}")

    # 비교 목록. 표의 열 이름으로 번호(1부터)를 쓴다
    print("\n[비교 목록] 번호, 종류, 비교 내용, 위치 조건이 참인 pair 수")
    for index, (family, image, (a_variant, a_slot), (b_variant, b_slot)) in enumerate(COMPARISONS):
        n_token = sum(row["same_token_index"] for row in by_index[index])
        n_position = sum(row["same_position_ids"] for row in by_index[index])
        print(f"{index + 1:>3}  {family:<13} {image}: {a_variant} 자리 {a_slot} vs {b_variant} 자리 {b_slot}"
              f"   토큰 인덱스 같음 {n_token}, position id 같음 {n_position}")

    # 기준선: 앞부분이 완전히 같은 비교. 여기가 0에 가깝지 않으면 다른 비교의 값에도 계산 오차가 섞여 있다
    baseline = same_prefix_baseline(rows)
    print(f"\n[기준선] same_prefix의 Relative L2 (pair 중앙값을 층별로 본 최댓값): {baseline:.3g}")
    if baseline >= SAME_WITHIN:
        print(f"  주의: 기준 {SAME_WITHIN}보다 크다. 이 실행의 사진 비교에는 계산 오차가 섞여 있어 아래 표를 그대로 해석할 수 없다")

    numbered = {index + 1: group for index, group in by_index.items()}
    print_layer_table("1. 토큰별 유사도: 토큰끼리 cosine을 구한 뒤 평균 (pair 중앙값)",
                      {number: median_by_layer(group, "token_cosine") for number, group in numbered.items()}, ".4f")
    print_layer_table("2. Relative L2: 차이의 크기 / 원래 크기 (pair 중앙값)",
                      {number: median_by_layer(group, "relative_l2") for number, group in numbered.items()}, ".4f")
    print_layer_table("3. 평균끼리의 유사도: 토큰을 먼저 평균 낸 벡터끼리의 cosine (pair 중앙값)",
                      {number: median_by_layer(group, "cosine_of_means") for number, group in numbered.items()}, ".4f")

    # 위치를 통제한 pair만: 토큰 인덱스와 position id가 모두 같은데도 차이가 남는가
    print("\n[위치를 통제한 비교] 번호 뒤에 *가 붙은 열은 토큰 인덱스와 position id가 모두 같은 pair만, 붙지 않은 열은 나머지 pair")
    columns = {}
    for index in ONE_IMAGE_CHANGED:
        controlled = [row for row in by_index[index] if is_position_controlled(row)]
        others = [row for row in by_index[index] if not is_position_controlled(row)]
        print(f"  {index + 1}: 통제된 pair {len(controlled)}개, 나머지 {len(others)}개")
        if controlled:
            columns[f"{index + 1}*"] = median_by_layer(controlled, "token_cosine")
        if others:
            columns[f"{index + 1}"] = median_by_layer(others, "token_cosine")
    print_layer_table("4. 토큰별 유사도, 위치를 통제한 pair와 나머지 (pair 중앙값)", columns, ".4f")


if __name__ == "__main__":
    main()
