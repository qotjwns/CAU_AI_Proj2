"""
저장된 pair 기록에서 사진 비교 값을 모아 층별로 요약한다

pair 기록(pairs/<pair>.pt)에는 비교 12개가 들어 있다 (mivlm.analysis.pair_record.COMPARISONS의 순서).
여기서는 실행 폴더 전체를 읽어 비교마다 한 행으로 펼치고, 여러 pair의 값을 층마다 하나로 줄인다.
"""
from pathlib import Path

import torch

from mivlm.analysis.pair_record import COMPARISONS
from mivlm.evaluation.metrics import read_results

# 위치를 통제한 pair만 따로 볼 비교 (COMPARISONS에서의 순번, 0부터). 모두 "자리는 같고 앞의 이미지 하나만 바뀐" 비교다
ONE_IMAGE_CHANGED = [3, 4, 6, 7, 8, 9]


def load_comparison_rows(run_dir: Path) -> list[dict]:
    """실행 폴더의 pair 기록을 모두 읽어 비교마다 한 행으로 만든다 (pair 400개면 4,800행, results.jsonl의 순서)

    행의 내용:
        pair: pair 이름
        index: COMPARISONS에서의 순번 (0부터)
        family, image: 비교의 종류와 비교하는 이미지 이름
        same_token_index, same_position_ids: 두 입력에서 토큰 인덱스, position id가 같은가
        token_cosine: 토큰별 유사도를 층마다 평균 낸 값, shape (층 수 + 1,)
        cosine_of_means, relative_l2: shape (층 수 + 1,)
    토큰별 값은 여기서 평균으로 줄인다. 토큰 단위로 보려면 .pt 파일을 직접 읽는다.
    """
    rows = []
    for result in read_results(run_dir):
        record = torch.load(run_dir / result["file"], map_location="cpu", weights_only=True)
        for index, comparison in enumerate(record["comparisons"]):
            rows.append({
                "pair": record["pair"],
                "index": index,
                "family": comparison["family"],
                "image": comparison["image"],
                "same_token_index": comparison["same_token_index"],
                "same_position_ids": comparison["same_position_ids"],
                "token_cosine": comparison["token_cosine"].mean(dim=1),
                "cosine_of_means": comparison["cosine_of_means"],
                "relative_l2": comparison["relative_l2"],
            })
    return rows


def median_by_layer(rows: list[dict], metric: str) -> torch.Tensor:
    """여러 행의 metric을 층마다 중앙값으로 줄인다: shape (층 수 + 1,)

    metric: 행에서 꺼낼 값의 이름. "token_cosine", "cosine_of_means", "relative_l2"
    평균이 아니라 중앙값을 쓰는 것은 값이 유난히 큰 pair 몇 개에 끌려가지 않게 하려는 것이다.
    행이 짝수 개면 가운데 두 값의 평균이다.
    """
    return torch.stack([row[metric] for row in rows]).quantile(0.5, dim=0)


def is_position_controlled(row: dict) -> bool:
    """비교에서 위치가 통제됐는가: 두 입력에서 그 이미지의 토큰 인덱스와 position id가 모두 같다

    이 조건에서 남는 차이는 위치가 밀려서 생긴 것이 아니다. 다만 앞에 놓인 이미지의 내용이 다르므로,
    차이가 그 내용 때문이라고 볼 근거가 될 뿐 그것만으로 원인이 확정되지는 않는다.
    """
    return row["same_token_index"] and row["same_position_ids"]


def same_prefix_baseline(rows: list[dict]) -> float:
    """기준선: same_prefix 비교(앞부분이 완전히 같은 두 입력)의 Relative L2. pair 중앙값을 층마다 구한 것 중 최댓값

    이 값이 0에 가깝지 않으면(SAME_WITHIN 이상) 그 실행의 다른 비교에도 계산 오차가 섞여 있어서 그대로 해석할 수 없다.
    """
    (index,) = [index for index, (family, *_) in enumerate(COMPARISONS) if family == "same_prefix"]
    return median_by_layer([row for row in rows if row["index"] == index], "relative_l2").max().item()
