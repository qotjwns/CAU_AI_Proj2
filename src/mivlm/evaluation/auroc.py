"""
AUROC: 점수가 두 묶음의 케이스를 순서대로 가르는 정도
"""
import torch


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """레이블이 1인 케이스 하나와 0인 케이스 하나를 뽑았을 때 1인 쪽의 점수가 더 높을 확률 (같으면 반으로 센다)

    1이면 완전히 갈리고 0.5면 구분하지 못한다. 예측을 가르는 기준점과 두 레이블의 케이스 수 비율에 좌우되지 않는다.
    scores: shape (..., 케이스 수). 앞의 차원(예: 층)이 있으면 그 차원마다 따로 계산한다
    labels: 0 또는 1, shape (케이스 수,)
    반환: shape (...). 한쪽 레이블의 케이스가 없으면 nan
    """
    positive = scores[..., labels == 1].unsqueeze(-1)
    negative = scores[..., labels == 0].unsqueeze(-2)
    return (positive > negative).float().mean(dim=(-2, -1)) + 0.5 * (positive == negative).float().mean(dim=(-2, -1))
