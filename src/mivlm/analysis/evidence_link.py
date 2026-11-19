"""
단일 이미지 증거와 공동 판단의 연계: 이미지를 한 장씩 물은 margin으로, 세 장을 같이 물은 margin을 예측해 본다

입력: A(세 장을 같이 물음, 01_baseline)의 행과 B(한 장씩 물음, 02_decomposition)를 케이스 단위로 맞춘 행.
    둘 다 mivlm.evaluation.decomposition.load_case_rows가 같은 케이스 순서로 돌려준다.

두 질문은 문장이 다르다 ("이 이미지에 있는가"와 "세 이미지 모두에 있는가"). 그래서 단일 margin과 공동 margin을
직접 빼거나 크기를 비교하지 않는다. 단일 margin은 학습 클래스에서 맞춘 계수를 거쳐서만 공동 margin의 눈금으로 옮긴다.

관측 (주 분석): (pair, 변형) 하나. 변형은 PPN, PNP, NPP
    target = 공동 margin(PPP) - 공동 margin(변형). P를 N으로 바꿨을 때의 하락이다
    min    = min(PPP의 단일 margin 3개) - min(변형의 단일 margin 3개)
    mean   = mean(PPP의 단일 margin 3개) - mean(변형의 단일 margin 3개) = (바뀐 P의 단일 margin - N의 단일 margin) / 3
    n      = -(N의 단일 margin). min과 mean이 공유하는 부분이다
관측 (보조 분석): 케이스 하나 (PPP와 세 변형). target = 공동 margin, min과 mean은 그 케이스의 단일 margin 3개의 값

모델: target = (절편 또는 자리마다의 절편) + 기울기 x (설명 변수 하나). DROP_MODELS, LEVEL_MODELS 참고.
평가: 클래스(또는 클래스의 덩어리) 하나를 빼고 나머지에서 계수를 맞춘 뒤, 뺀 쪽을 예측한다. 계수는 학습 쪽에서만 정한다.

여기서 비교하는 것은 예측 규칙의 정확도다. min을 쓴 규칙이 더 정확해도 "모델이 내부에서 최솟값을 계산한다"는 뜻이 아니다.
"""
import statistics

import torch

from mivlm.evaluation.metrics import ONE_N_VARIANTS

# 모델 이름 -> (설명 변수의 키 또는 None, 자리마다 절편을 따로 두는가)
DROP_MODELS = {
    "intercept": (None, False), "slot": (None, True),
    "n": ("n", False), "min": ("min", False), "mean": ("mean", False),
    "n+slot": ("n", True), "min+slot": ("min", True), "mean+slot": ("mean", True),
}
# 보조 분석에는 PPP 케이스도 들어가서 N만 쓰는 모델이 없다. "자리"는 변형(PPP, PPN, PNP, NPP)이다
LEVEL_MODELS = {name: model for name, model in DROP_MODELS.items() if model[0] != "n"}
DELTA_SHARE = 0.05  # 실질적 차이의 기준: 절편만 쓴 모델의 오차의 이 비율. 연구자가 정한 값이다


def with_single_margins(a_rows: list[dict], b_rows: list[dict]) -> list[dict]:
    """A의 행에 그 케이스 이미지 3장의 단일 margin(자리 순서)을 single_margins로 붙인다

    a_rows, b_rows: load_case_rows의 case_rows["A"], case_rows["B"]. 케이스 순서가 다르면 에러를 낸다.
    """
    if [row["case_id"] for row in a_rows] != [row["case_id"] for row in b_rows]:
        raise ValueError("A와 B의 케이스 순서가 다름")
    return [{**a_row, "single_margins": b_row["image_margins"]} for a_row, b_row in zip(a_rows, b_rows)]


def drop_observations(pairs: list[dict]) -> list[dict]:
    """주 분석의 관측: pair마다 변형 셋에 하나씩. pair의 행에는 single_margins가 붙어 있어야 한다

    관측마다: object, slot(변형 이름), target, min, mean, n (파일 맨 위의 정의),
    n_image와 n_single (N 이미지와 그 단일 margin), n_is_lowest (N의 단일 margin이 세 장 중 가장 낮은가),
    false_yes (변형 케이스에 모델이 Yes로 답했는가)
    """
    observations = []
    for pair in pairs:
        ppp = pair["cases"]["PPP"]
        for variant in ONE_N_VARIANTS:
            changed = pair["cases"][variant]
            n_slot = variant.index("N")
            single = changed["single_margins"]
            observations.append({
                "object": ppp["object"],
                "slot": variant,
                "target": ppp["logit_margin"] - changed["logit_margin"],
                "min": min(ppp["single_margins"]) - min(single),
                "mean": statistics.mean(ppp["single_margins"]) - statistics.mean(single),
                "n": -single[n_slot],
                "n_image": changed["image_ids"][n_slot],
                "n_single": single[n_slot],
                "n_is_lowest": single[n_slot] == min(single),
                "false_yes": changed["prediction"] == "Yes",
            })
    return observations


def level_observations(pairs: list[dict]) -> list[dict]:
    """보조 분석의 관측: pair의 케이스(PPP와 세 변형)마다 하나씩. target은 그 케이스의 공동 margin이다"""
    return [{"object": case["object"], "slot": variant, "target": case["logit_margin"],
             "min": min(case["single_margins"]), "mean": statistics.mean(case["single_margins"])}
            for pair in pairs for variant, case in pair["cases"].items()]


def out_of_fold_predictions(observations: list[dict], fold_of: dict, models: dict) -> dict[str, list[float]]:
    """모델마다, 관측마다의 예측값: {모델 이름: [예측값, ...]} (observations와 같은 순서)

    fold_of: {클래스: 묶음 이름}. 관측이 속한 묶음을 통째로 빼고, 나머지 관측으로 최소제곱 계수를 맞춰 예측한다.
        클래스를 하나씩 빼려면 {클래스: 클래스}, 덩어리로 빼려면 {클래스: 덩어리 번호}를 준다
    models: {모델 이름: (설명 변수의 키 또는 None, 자리마다 절편을 따로 두는가)}
    뺀 묶음의 target은 그 묶음의 예측에 쓰이지 않는다.
    """
    levels = sorted({observation["slot"] for observation in observations})
    folds = [fold_of[observation["object"]] for observation in observations]
    target = torch.tensor([observation["target"] for observation in observations], dtype=torch.float64)
    predictions = {}
    for name, (predictor, by_slot) in models.items():
        # 열: 자리마다의 절편(그 자리면 1) 또는 공통 절편 하나, 그리고 설명 변수가 있으면 그 값
        rows = [[float(observation["slot"] == level) for level in levels] if by_slot else [1.0]
                for observation in observations]
        if predictor:
            rows = [row + [observation[predictor]] for row, observation in zip(rows, observations)]
        x = torch.tensor(rows, dtype=torch.float64)
        predicted = torch.empty(len(observations), dtype=torch.float64)
        for fold in set(folds):
            held_out = torch.tensor([observation_fold == fold for observation_fold in folds])
            coefficients = torch.linalg.lstsq(x[~held_out], target[~held_out].unsqueeze(1)).solution[:, 0]
            predicted[held_out] = x[held_out] @ coefficients
        predictions[name] = predicted.tolist()
    return predictions


def class_mean(observations: list[dict], values: list[float]) -> dict[str, float]:
    """관측마다의 값을 클래스별로 평균한다: {클래스: 평균} (클래스가 처음 나온 순서)"""
    by_class = {}
    for observation, value in zip(observations, values):
        by_class.setdefault(observation["object"], []).append(value)
    return {name: statistics.mean(group) for name, group in by_class.items()}


def n_strength_bins(observations: list[dict]) -> list[int]:
    """관측마다, 그 N 이미지가 클래스 안에서 어느 구간인가: 0, 1, 2

    클래스의 N 이미지를 단일 margin이 낮은 순서(더 강하게 No)로 세우고, 이미지 수로 셋으로 나눈다.
    0은 가장 강하게 No인 3분의 1, 2는 가장 약하게 No인 3분의 1이다. 이미지가 8장이면 3, 3, 2장으로 나뉜다.
    """
    single_of = {}  # 클래스 -> {N 이미지: 단일 margin}
    for observation in observations:
        single_of.setdefault(observation["object"], {})[observation["n_image"]] = observation["n_single"]
    bin_of = {}
    for name, singles in single_of.items():
        ordered = sorted(singles, key=lambda image_id: (singles[image_id], image_id))
        for rank, image_id in enumerate(ordered):
            bin_of[(name, image_id)] = rank * 3 // len(ordered)
    return [bin_of[(observation["object"], observation["n_image"])] for observation in observations]


def practical_verdict(low: float, high: float, delta: float) -> str:
    """두 모델의 오차 차이(뒤 모델의 오차 - 앞 모델의 오차)의 구간으로 내리는 실질적 효과의 판정

    구간 전체가 +delta를 넘으면 앞 모델이, 전체가 -delta 아래면 뒤 모델이 실질적으로 우월하다.
    구간 전체가 -delta ~ +delta 안이면 실질적 차이가 없다. 그 밖에는 미결정이다.
    구간이 0을 제외하는지(통계적 방향)는 따로 본다 (mivlm.analysis.class_sensitivity.direction_verdict).
    """
    if low > delta:
        return "앞 모델이 실질적으로 우월"
    if high < -delta:
        return "뒤 모델이 실질적으로 우월"
    if -delta <= low and high <= delta:
        return "실질적 차이 없음"
    return "미결정"
