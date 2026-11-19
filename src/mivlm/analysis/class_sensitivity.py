"""
클래스 내부 민감도: 같은 클래스 안에서 PPP의 한 자리를 N 이미지로 바꾸면 답과 logit margin이 얼마나 달라지는가

입력은 A(01_baseline) 실행의 행이다 (mivlm.evaluation.metrics.load_rows). PPP와 N이 1장인 변형(PPN, PNP, NPP)만 쓴다.
같은 pair의 PPN, PNP, NPP는 그 pair의 PPP에서 한 자리만 같은 N 이미지로 바꾼 것이다. 그래서 변화는 pair 안에서 짝지어 잰다.

값을 읽는 방향
    margin 하락 = margin(PPP) - margin(N을 넣은 변형). 양수면 N을 넣었을 때 No 쪽으로 움직인 것이다
    AUROC = 같은 클래스의 PPP 케이스와 N을 넣은 케이스를 하나씩 뽑았을 때 PPP의 margin이 더 높을 확률 (같으면 반으로 센다)
    전환 = (PPP에서의 답)->(N을 넣은 변형에서의 답). Yes->No는 맞게 바뀐 것, Yes->Yes는 False-Yes로 남은 것이다

같은 이미지를 여러 케이스가 돌려 쓰므로 케이스는 서로 독립이 아니다
    클래스들의 평균과 상관: 클래스를 단위로 다시 뽑아 구간을 만든다 (class_bootstrap_interval)
    클래스 하나의 값: 구간을 내지 않는다. 이미지 한 장을 뺐을 때 값이 움직이는 범위만 낸다 (leave_one_image_out)
    두 클래스가 같이 쓰는 이미지: 그 이미지가 든 pair를 빼고 다시 계산해 본다 (shared_images, without_images)
"""
import statistics
from collections import Counter

import torch

from mivlm.data.cases import case_pairs
from mivlm.evaluation.auroc import auroc
from mivlm.evaluation.metrics import ONE_N_VARIANTS, summarize

ONE_N = "1N"  # N이 1장인 세 변형(PPN, PNP, NPP)을 합친 것의 이름
TRANSITIONS = ["Yes->No", "Yes->Yes", "No->No", "No->Yes"]  # (PPP에서의 답)->(N을 넣은 변형에서의 답)
N_DRAWS = 10_000  # 클래스를 다시 뽑는 횟수
SEED = 0  # 다시 뽑는 순서를 정하는 seed. 고정해서 같은 입력이면 같은 구간이 나오게 한다
# 분석에서 실행마다 맡는 역할. 주 분석의 판정이 기준이고, 나머지는 같은 판정이 나오는지를 본다
RUN_ROLES = {"qwen2.5-vl-3b": "주 분석", "qwen2.5-vl-7b": "모델 견고성 분석", "qwen2.5-vl-3b_float32": "정밀도 민감도 분석"}
NO_DIFFERENCE = (0.45, 0.55)  # 클래스 내부 AUROC의 평균이 이 범위 안이면 "순위로는 실질적으로 구분하지 못한다"로 본다


def class_pairs(rows: list[dict]) -> dict[str, list[dict]]:
    """행을 클래스별 pair 목록으로 묶는다: {클래스: [mivlm.data.cases.case_pairs의 pair, ...]}

    PPP와 N이 1장인 변형의 행만 쓴다. pair마다 {"pair": 이름, "cases": {"PPP": 행, "PPN": 행, "PNP": 행, "NPP": 행}}이다.
    """
    pairs_by_class = {}
    for pair in case_pairs([row for row in rows if row["variant"] in ["PPP", *ONE_N_VARIANTS]]):
        pairs_by_class.setdefault(pair["cases"]["PPP"]["object"], []).append(pair)
    return pairs_by_class


def class_values(pairs: list[dict]) -> dict[str, float]:
    """pair들(보통 한 클래스의 pair 전부)로 계산한 값. 이름 끝의 PPN, PNP, NPP는 그 변형, 1N은 세 변형을 합친 것이다

    n_pairs                      pair 수. 변형마다의 케이스 수와 같다 (1N은 그 3배)
    yes_rate_*, mean_margin_*    그 변형에서 Yes로 답한 비율, margin의 평균 (PPP 포함)
    yes_rate_drop_*              PPP의 Yes 비율 - 그 변형의 Yes 비율. Yes->No 비율에서 No->Yes 비율을 뺀 것과 같다
    margin_drop_*                같은 pair 안에서 margin(PPP) - margin(그 변형)의 평균
    margin_drop_positive_rate_*  그 차이가 0보다 큰 pair의 비율
    Yes->No_* 등                 전환 비율 (TRANSITIONS). 판정 불가 답이 있으면 네 비율의 합이 1보다 작다
    auroc_*                      PPP 케이스와 그 변형의 케이스를 하나씩 뽑았을 때 PPP의 margin이 더 높을 확률
    """
    ppp = [pair["cases"]["PPP"] for pair in pairs]
    ppp_summary = summarize(ppp)
    values = {"n_pairs": len(pairs), "yes_rate_PPP": ppp_summary["yes_rate"],
              "mean_margin_PPP": ppp_summary["mean_margin"]}
    for name, variants in [*[(variant, [variant]) for variant in ONE_N_VARIANTS], (ONE_N, ONE_N_VARIANTS)]:
        changes = [(pair["cases"]["PPP"], pair["cases"][variant]) for pair in pairs for variant in variants]
        changed_summary = summarize([changed for _, changed in changes])
        drops = [before["logit_margin"] - changed["logit_margin"] for before, changed in changes]
        transitions = Counter(f"{before['prediction']}->{changed['prediction']}" for before, changed in changes)
        margins = torch.tensor([row["logit_margin"] for row in ppp + [changed for _, changed in changes]],
                               dtype=torch.float64)
        is_ppp = torch.tensor([1] * len(ppp) + [0] * len(changes))
        values |= {
            f"yes_rate_{name}": changed_summary["yes_rate"],
            f"mean_margin_{name}": changed_summary["mean_margin"],
            f"yes_rate_drop_{name}": ppp_summary["yes_rate"] - changed_summary["yes_rate"],
            f"margin_drop_{name}": statistics.mean(drops),
            f"margin_drop_positive_rate_{name}": sum(drop > 0 for drop in drops) / len(drops),
            **{f"{transition}_{name}": transitions[transition] / len(changes) for transition in TRANSITIONS},
            f"auroc_{name}": auroc(margins, is_ppp).item(),
        }
    return values


def pair_images(pair: dict) -> set[int]:
    """pair의 네 케이스(PPP, PPN, PNP, NPP)에 나오는 이미지 전부"""
    return {image_id for case in pair["cases"].values() for image_id in case["image_ids"]}


def shared_images(pairs_by_class: dict[str, list[dict]]) -> set[int]:
    """두 개 이상의 클래스에 나오는 이미지"""
    classes_of_image = {}
    for name, pairs in pairs_by_class.items():
        for pair in pairs:
            for image_id in pair_images(pair):
                classes_of_image.setdefault(image_id, set()).add(name)
    return {image_id for image_id, names in classes_of_image.items() if len(names) > 1}


def without_images(pairs: list[dict], image_ids: set[int]) -> list[dict]:
    """image_ids의 이미지가 하나라도 든 pair를 통째로 뺀다. PPP와 N을 넣은 변형의 짝이 깨지지 않게 pair 단위로 뺀다"""
    return [pair for pair in pairs if not pair_images(pair) & image_ids]


def leave_one_image_out(pairs: list[dict], values_of) -> dict[str, tuple[float, float]]:
    """이미지 한 장을 뺄 때마다 값을 다시 계산해, 값마다 (최솟값, 최댓값)을 돌려준다

    values_of: pair 목록을 받아 {이름: 값}을 돌려주는 함수. 예: class_values
    그 pair들에 나오는 이미지마다, 그 이미지가 든 pair를 통째로 빼고 계산한다. 뺐더니 pair가 남지 않는 이미지는 건너뛴다.
    구간이 아니다. 이미지 한 장에 값이 얼마나 좌우되는지를 보는 범위다.
    """
    images = sorted(set().union(*(pair_images(pair) for pair in pairs)))
    recomputed = [values_of(kept) for image_id in images if (kept := without_images(pairs, {image_id}))]
    return {key: (min(values[key] for values in recomputed), max(values[key] for values in recomputed))
            for key in recomputed[0]}


def class_bootstrap_interval(values: torch.Tensor, statistic, level: float) -> torch.Tensor:
    """클래스를 단위로 다시 뽑아 만든 백분위 구간: shape (2, ...)의 [하한, 상한]

    values: 클래스마다의 값, shape (클래스 수, ...). 클래스 수만큼을 복원 추출로 N_DRAWS번 뽑는다
    statistic: 뽑은 값 (N_DRAWS, 클래스 수, ...)를 받아 뽑기마다의 통계량 (N_DRAWS, ...)를 돌려주는 함수
    level: 구간의 수준. 0.95면 통계량의 2.5% 지점과 97.5% 지점
    통계량이 nan인 뽑기(뽑힌 값이 모두 같아 상관을 정할 수 없는 경우)는 빼고 계산한다.
    뽑는 순서는 SEED로 고정되어 있어서, 클래스 수가 같으면 어떤 통계량이든 같은 클래스들을 뽑는다.
    단위인 클래스 수가 적으면 이 구간은 실제보다 조금 좁을 수 있다.
    """
    picked = torch.randint(len(values), (N_DRAWS, len(values)), generator=torch.Generator().manual_seed(SEED))
    draws = statistic(values[picked])
    tail = (1 - level) / 2
    return torch.nanquantile(draws, torch.tensor([tail, 1 - tail], dtype=draws.dtype), dim=0)


def mean_interval(per_class: list[float], level: float = 0.95) -> tuple[float, float, float]:
    """클래스마다의 값의 평균과 그 구간: (평균, 하한, 상한). 모든 클래스가 같은 무게를 갖는다"""
    values = torch.tensor(per_class, dtype=torch.float64)
    low, high = class_bootstrap_interval(values, lambda drawn: drawn.mean(dim=1), level).tolist()
    return values.mean().item(), low, high


def rank_correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """순위 상관(Spearman): x, y shape (..., n) -> shape (...). 값이 같으면 평균 순위를 준다

    한쪽의 값이 모두 같으면 순위에 차이가 없어 상관을 정할 수 없고, nan이 나온다.
    """
    def centered_ranks(v: torch.Tensor) -> torch.Tensor:
        smaller = (v.unsqueeze(-1) > v.unsqueeze(-2)).sum(dim=-1)  # 자기보다 작은 값의 수
        equal = (v.unsqueeze(-1) == v.unsqueeze(-2)).sum(dim=-1)  # 자기와 같은 값의 수 (자기 포함)
        ranks = smaller + (equal + 1) / 2
        return ranks - ranks.mean(dim=-1, keepdim=True)

    rx, ry = centered_ranks(x), centered_ranks(y)
    return (rx * ry).sum(dim=-1) / (rx.square().sum(dim=-1) * ry.square().sum(dim=-1)).sqrt()


def rank_correlation_interval(x: list[float], y: list[float], level: float = 0.95) -> tuple[float, float, float]:
    """클래스마다의 두 값 사이의 순위 상관과 그 구간: (상관, 하한, 상한)"""
    values = torch.tensor([x, y], dtype=torch.float64).T  # (클래스 수, 2)
    low, high = class_bootstrap_interval(values, lambda drawn: rank_correlation(drawn[..., 0], drawn[..., 1]),
                                         level).tolist()
    return rank_correlation(values[:, 0], values[:, 1]).item(), low, high


def explained_variance(rows: list[dict], keys: list[str]) -> float:
    """margin의 분산 중, keys(예: ["object"], ["object", "variant"])의 값이 같은 행끼리 묶은 묶음의 평균이 설명하는 몫"""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row["logit_margin"])
    overall = statistics.mean(row["logit_margin"] for row in rows)
    within = sum((margin - statistics.mean(group)) ** 2 for group in groups.values() for margin in group)
    return 1 - within / sum((row["logit_margin"] - overall) ** 2 for row in rows)


def direction_verdict(low: float, high: float) -> str:
    """"값이 0보다 크다"는 주장의 판정: 구간 전체가 0보다 크면 지지, 전체가 0보다 작으면 반증, 0을 포함하면 미결정"""
    if low > 0:
        return "지지"
    if high < 0:
        return "반증"
    return "미결정"


def difference_verdict(low: float, high: float) -> str:
    """"두 값이 다르다"는 주장의 판정: 차이의 구간이 0을 제외하면 지지, 아니면 미결정 ("같다"의 판정 기준은 두지 않는다)"""
    return "지지" if low > 0 or high < 0 else "미결정"


def ranking_verdict(low: float, high: float) -> str:
    """클래스 내부 AUROC 평균의 구간으로 내리는 판정

    구간 전체가 NO_DIFFERENCE 안이면 "무차이 지지", 하한이 그 범위의 위쪽 끝을 넘으면 "무차이 기각", 그 밖에는 "미결정"
    클래스 내부의 순위 구분력에 대한 판정일 뿐이다. "오류가 클래스 성향만으로 설명된다"의 증거로 읽지 않는다.
    """
    if NO_DIFFERENCE[0] <= low and high <= NO_DIFFERENCE[1]:
        return "무차이 지지"
    if low > NO_DIFFERENCE[1]:
        return "무차이 기각"
    return "미결정"
