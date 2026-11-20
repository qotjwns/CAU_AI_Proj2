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

자리 가중치와 하락 몫 (weight_terms, weight_statistics). pair의 행에는 single_margins(이미지 3장의 단일 margin)가 있어야 한다
    자리 k(1, 2, 3)는 N이 들어간 자리다. 자리 k의 변형은 SLOT_VARIANTS[k - 1]이다 (자리 1은 NPP, 2는 PNP, 3은 PPN)
    w      PPP의 공동 margin을 단일 margin 3개로 클래스 내부 회귀한 계수: (Σ_c X_cᵀX_c)⁻¹ Σ_c X_cᵀy_c
           X_c는 클래스 평균을 뺀 단일 margin (pair 수, 3), y_c는 클래스 평균을 뺀 PPP margin
    D_k    공동 margin의 하락 m(PPP) - m(자리 k에 N)의 클래스 평균을, 클래스들에 대해 평균한 값
    Δ_k    단일 margin의 변화 s(바뀐 P) - s(N)의 클래스 평균을, 클래스들에 대해 평균한 값
    δ_k = D_k / ΣD (관측 하락 몫),  δ̂_k = w_kΔ_k / Σ w_jΔ_j (가산 모형의 예측 몫),  r_k = δ_k - δ̂_k (주 지표, 합이 0)
    보조: ω_k = w_k / Σw,  δ_k - ω_k,  ρ_k = D_k / (w_kΔ_k),  기울기 셋 (weight_statistics 참고)
    w는 클래스 합으로, D와 Δ는 클래스 평균으로 무게를 준다. 단일 질문과 공동 질문은 문장이 달라서, 단일 margin은
    계수 w를 거쳐서만 공동 margin의 눈금으로 옮긴다.
"""
import torch

from mivlm.evaluation.metrics import ONE_N_VARIANTS

SUM_NAMES = ["n", "x", "y", "xx", "xy"]  # slope_sums의 마지막 차원의 순서: [n, Σx, Σy, Σx², Σxy]
SLOT_VARIANTS = sorted(ONE_N_VARIANTS, key=lambda variant: variant.index("N"))  # 자리 1, 2, 3의 변형: NPP, PNP, PPN
# weight_terms의 마지막 차원의 순서. 자리 j마다:
#   xx1~xx3   XᵀX의 j행,  xy  (Xᵀy)_j,  yy  yᵀy (자리마다 같은 값),  y2  Σ m(PPP)² (yy의 기준 크기, 자리마다 같은 값)
#   drop      D_j의 클래스 값,  change  Δ_j의 클래스 값
#   variant   Σ (m(자리 j에 N) - 평균)(m(PPP) - 평균),  n_single  Σ (s(N) - 평균)(m(PPP) - 평균)
TERM_NAMES = ["xx1", "xx2", "xx3", "xy", "yy", "y2", "drop", "change", "variant", "n_single"]
RELATIVE_TOLERANCE = 1e-12  # 분모가 그 기준 크기의 이 배 이하면 0으로 보고, 값을 정하지 않는다(nan)


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


def weight_terms(pairs_by_class: dict[str, list[dict]]) -> torch.Tensor:
    """클래스마다, 자리마다 가중치와 몫의 항: shape (클래스 수, 자리 3개, 10), float64

    클래스 순서는 pairs_by_class의 순서, 자리 순서는 1, 2, 3, 마지막 차원은 TERM_NAMES다.
    클래스 합으로 더할 항(XᵀX, Xᵀy, yᵀy, Σm², variant, n_single)과 클래스 평균으로 모을 항(drop, change)이 섞여 있다.
    """
    per_class = []
    for pairs in pairs_by_class.values():
        ppp = [pair["cases"]["PPP"] for pair in pairs]
        single = torch.tensor([case["single_margins"] for case in ppp], dtype=torch.float64)  # (pair 수, 자리 3개)
        margin = torch.tensor([case["logit_margin"] for case in ppp], dtype=torch.float64)  # (pair 수,)
        changed = [[pair["cases"][variant] for variant in SLOT_VARIANTS] for pair in pairs]
        changed_margin = torch.tensor([[case["logit_margin"] for case in row] for row in changed], dtype=torch.float64)
        n_single = torch.tensor([[case["single_margins"][slot] for slot, case in enumerate(row)] for row in changed],
                                dtype=torch.float64)  # 자리 j에 들어간 N의 단일 margin
        x = single - single.mean(dim=0)
        y = margin - margin.mean()
        yy = torch.full((3, 1), (y * y).sum().item(), dtype=torch.float64)
        y2 = torch.full((3, 1), (margin * margin).sum().item(), dtype=torch.float64)
        columns = [
            x.T @ x,  # (3, 3)
            (x.T @ y).unsqueeze(1),
            yy, y2,
            (margin.unsqueeze(1) - changed_margin).mean(dim=0).unsqueeze(1),
            (single - n_single).mean(dim=0).unsqueeze(1),
            ((changed_margin - changed_margin.mean(dim=0)).T @ y).unsqueeze(1),
            ((n_single - n_single.mean(dim=0)).T @ y).unsqueeze(1),
        ]
        per_class.append(torch.cat(columns, dim=1))
    return torch.stack(per_class)


def shares(parts: torch.Tensor) -> torch.Tensor:
    """마지막 축의 몫: parts / parts의 합. 합이 0 이하이거나 0에 가까우면 nan

    0에 가깝다: 합이 |parts|의 합의 RELATIVE_TOLERANCE배 이하. 부호가 섞여 합이 거의 상쇄된 경우도 여기에 든다.
    """
    total = parts.sum(dim=-1, keepdim=True)
    scale = parts.abs().sum(dim=-1, keepdim=True)
    return torch.where(total > RELATIVE_TOLERANCE * scale, parts / total, torch.nan)


def weight_statistics(terms: torch.Tensor) -> dict[str, torch.Tensor]:
    """weight_terms의 항 (..., 클래스 수, 자리 3개, 10)으로 낸 통계량: {이름: (..., 자리 3개)}

    weights          w. XᵀX가 특이하거나 조건수가 1 / RELATIVE_TOLERANCE를 넘으면 nan
    weight_share     ω_k = w_k / Σw (shares)
    drop, change     D_k, Δ_k
    drop_share       δ_k = D_k / ΣD (shares)
    predicted_share  δ̂_k = w_kΔ_k / Σ w_jΔ_j (shares)
    share_gap        r_k = δ_k - δ̂_k. 주 지표
    share_vs_weight  δ_k - ω_k
    magnitude_ratio  ρ_k = D_k / (w_kΔ_k). 분모는 음수여도 된다. |w_kΔ_k|가 Σ|w_jΔ_j|의 RELATIVE_TOLERANCE배 이하면 nan
    kept_slope       Σ_{j≠k} w_j Cov(s_j, m) / Var(m). 남은 두 자리의 P 증거만으로 낸 기울기
    additive_slope   (Σ_{j≠k} w_j Cov(s_j, m) + w_k Cov(s_N, m)) / Var(m). N의 단일 증거까지 넣은 가산 모형의 기울기
    observed_slope   Cov(m(자리 k에 N), m) / Var(m). 7-A-1의 클래스 내부 기울기와 같은 값이다
    slope_gap        observed_slope - additive_slope
    Cov, Var는 클래스 평균을 뺀 값의 곱의 합이고 m은 PPP margin이다. Var(m)이 Σm²의 RELATIVE_TOLERANCE배 이하면 기울기는 nan
    """
    summed = terms.sum(dim=-3)  # (..., 자리 3개, 10)
    averaged = terms.mean(dim=-3)
    xx, xy = summed[..., :, 0:3], summed[..., :, 3]
    yy, y2 = summed[..., 0, 4:5], summed[..., 0, 5:6]  # (..., 1): 자리 축으로 넓혀 쓴다
    drop, change = averaged[..., :, 6], averaged[..., :, 7]
    variant, n_single = summed[..., :, 8], summed[..., :, 9]

    solution, info = torch.linalg.solve_ex(xx, xy.unsqueeze(-1))
    unstable = (info != 0) | ~(torch.linalg.cond(xx) < 1 / RELATIVE_TOLERANCE)  # cond가 inf나 nan이어도 불안정으로 본다
    weights = torch.where(unstable.unsqueeze(-1), torch.nan, solution.squeeze(-1))

    predicted = weights * change
    drop_share, predicted_share, weight_share = shares(drop), shares(predicted), shares(weights)
    ratio_scale = predicted.abs().sum(dim=-1, keepdim=True)
    magnitude_ratio = torch.where(predicted.abs() > RELATIVE_TOLERANCE * ratio_scale, drop / predicted, torch.nan)

    has_variance = yy > RELATIVE_TOLERANCE * y2
    weighted = weights * xy  # w_j Cov(s_j, m)
    kept = weighted.sum(dim=-1, keepdim=True) - weighted
    additive = kept + weights * n_single
    kept_slope, additive_slope, observed_slope = (torch.where(has_variance, value / yy, torch.nan)
                                                  for value in [kept, additive, variant])
    return {
        "weights": weights, "weight_share": weight_share, "drop": drop, "change": change,
        "drop_share": drop_share, "predicted_share": predicted_share, "share_gap": drop_share - predicted_share,
        "share_vs_weight": drop_share - weight_share, "magnitude_ratio": magnitude_ratio,
        "kept_slope": kept_slope, "additive_slope": additive_slope, "observed_slope": observed_slope,
        "slope_gap": observed_slope - additive_slope,
    }
