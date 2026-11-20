"""
자리별 증거가 공동 판단에 남는 정도: 같은 pair의 1N margin을 PPP margin에 회귀한 기울기

입력은 mivlm.analysis.class_sensitivity.class_pairs가 돌려주는 클래스별 pair 목록이다.
같은 pair의 PPN, PNP, NPP는 그 pair의 PPP에서 한 자리(각각 3, 2, 1번 자리)만 같은 N 이미지로 바꾼 것이다.

기울기 (변형마다 따로)
    x = pair의 PPP margin, y = 같은 pair의 변형 margin
    전체 회귀       모든 클래스의 pair를 한데 놓은 OLS 기울기. 클래스 사이의 차이(클래스 성향)도 들어간다
    클래스 내부 회귀  x와 y에서 그 클래스의 평균을 뺀 뒤의 OLS 기울기. 클래스 사이의 차이를 뺀다
    둘 다 충분통계 [n, Σx, Σy, Σx², Σxy]를 클래스마다 모아 두고, 클래스 축으로 더해서 계산한다.
    그래서 클래스를 다시 뽑는 구간(class_bootstrap_interval)을 같은 함수로 낼 수 있다.

읽는 법: 기울기는 PPP margin의 변동 중 1N margin에 남는 몫이다. 가산 결합이면 그 값은 남은 두 자리가 PPP 변동에서
차지하는 몫으로 정해지므로, 기울기만으로 AND 여부나 모델 내부의 집계 방식을 판단하지 않는다.
"""
import torch

from mivlm.evaluation.metrics import ONE_N_VARIANTS

SUM_NAMES = ["n", "x", "y", "xx", "xy"]  # slope_sums의 마지막 차원의 순서: [n, Σx, Σy, Σx², Σxy]


def slope_sums(pairs_by_class: dict[str, list[dict]], centered: bool) -> torch.Tensor:
    """클래스마다, 변형마다 기울기의 충분통계: shape (클래스 수, 변형 3개, 5), float64

    클래스 순서는 pairs_by_class의 순서, 변형 순서는 ONE_N_VARIANTS(PPN, PNP, NPP), 마지막 차원은 SUM_NAMES다.
    centered=True면 x와 y에서 그 클래스의 평균을 뺀 값으로 센다 (클래스 내부 회귀). 이때 Σx와 Σy는 0이다.
    """
    per_class = []
    for pairs in pairs_by_class.values():
        y = torch.tensor([[pair["cases"][variant]["logit_margin"] for variant in ONE_N_VARIANTS] for pair in pairs],
                         dtype=torch.float64)  # (pair 수, 변형 3개)
        x = torch.tensor([pair["cases"]["PPP"]["logit_margin"] for pair in pairs],
                         dtype=torch.float64).unsqueeze(1).expand_as(y)  # 같은 PPP margin을 변형마다 쓴다
        if centered:
            x = x - x.mean(dim=0)
            y = y - y.mean(dim=0)
        n = torch.full((len(ONE_N_VARIANTS),), float(len(pairs)), dtype=torch.float64)
        per_class.append(torch.stack([n, x.sum(dim=0), y.sum(dim=0), (x * x).sum(dim=0), (x * y).sum(dim=0)], dim=-1))
    return torch.stack(per_class)


def slopes(sums: torch.Tensor) -> torch.Tensor:
    """충분통계를 클래스 축으로 더해 구한 OLS 기울기: (..., 클래스 수, 변형 수, 5) -> (..., 변형 수)

    기울기 = (nΣxy - ΣxΣy) / (nΣx² - (Σx)²). x의 변동이 없으면(분모가 0 이하) nan이다.
    클래스를 다시 뽑은 값 (뽑기 수, 클래스 수, 변형 수, 5)를 넣으면 뽑기마다의 기울기가 나온다.
    """
    n, sum_x, sum_y, sum_xx, sum_xy = sums.sum(dim=-3).unbind(dim=-1)
    spread = n * sum_xx - sum_x * sum_x
    return torch.where(spread > 0, (n * sum_xy - sum_x * sum_y) / spread, torch.nan)
