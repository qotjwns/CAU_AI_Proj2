"""
결정 기준 이동(fixed logit bias): 모델은 그대로 두고, 답을 읽는 기준만 옮겼을 때 False-Yes와 PPP가 어떻게 맞바뀌는가

입력은 A(01_baseline) 실행의 행 중 PPP와 N이 1장인 변형의 행이다 (mivlm.evaluation.metrics.load_rows).

규칙: logit margin > b이면 Yes, 아니면 No. margin이 b와 같으면 No다.
    b = 0이 모델의 원래 답이다 (margin이 0보다 커야 Yes). b를 올리면 Yes가 줄어든다 (No 쪽으로의 이동).
주 분석은 No 쪽으로의 이동만 본다 (b >= 0, no_shift_candidates). b가 음수도 될 수 있는 양방향 보정
(two_way_candidates)은 별도 분석이다. 후보 b는 기준을 고르는 쪽 행들의 margin 값이고, 평가하는 쪽의 margin은 쓰지 않는다.

기준을 정하는 방법은 assign_biases 하나로 나타낸다 (selection_rules 참고)
    global        다른 클래스들에서 고른 기준 하나를 그 클래스에 적용한다. 실제로 쓸 수 있는 방법이다
    class_cv      같은 클래스의 다른 그룹에서 고른 기준. 그 클래스의 정답을 쓰고, 그룹끼리 이미지가 겹쳐서 낙관적이다
    class_oracle  같은 클래스의 모든 케이스를 보고 고른 기준. 방법이 아니라 상한이다
global과 class_cv의 차이는 쓰는 정보와 나누는 방식이 달라서 생긴 것이기도 하다. 클래스 성향의 인과적 효과로 읽지 않는다.

기준 이동은 모델을 고치는 것이 아니라 답을 읽는 방식을 바꾸는 것이다.
"""
import torch

DAMAGE_LIMITS = [0.02, 0.05, 0.10]  # PPP 손상률의 한도
PRACTICAL_DELTA = 0.02  # 실질적 효과의 기준: 균형 정확도 차이 2%p. 연구자가 정한 값이다


def no_shift_candidates(rows: list[dict]) -> list[float]:
    """No 쪽으로의 이동(b >= 0)의 후보 b: 0과, rows에 있는 양수 margin 값 전부 (작은 것부터, 값은 반올림하지 않는다)

    margin > b가 Yes이므로, rows의 답이 바뀌는 곳은 margin 값뿐이다. 이웃한 두 margin 사이의 b는 rows에서 결과가 모두 같고,
    후보는 그 구간의 아래쪽 끝(0에 가까운 쪽)이다. 그래서 고정 간격 격자와 달리 놓치는 기준이 없다.
    """
    return sorted({0.0} | {row["logit_margin"] for row in rows if row["logit_margin"] > 0})


def two_way_candidates(rows: list[dict]) -> list[float]:
    """양방향 보정의 후보 b: 0과, rows에 있는 margin 값 전부 (음수 포함, 작은 것부터)

    후보는 no_shift_candidates와 같이 구간의 아래쪽 끝이다. 그래서 음수 쪽에서는 0에서 가장 먼 쪽이 후보가 된다.
    """
    return sorted({0.0} | {row["logit_margin"] for row in rows})


def _rate(count: int, total: int) -> float:
    """count / total. 분모가 0이면 nan (해당하는 케이스가 없어 비율을 정할 수 없다)"""
    return count / total if total else float("nan")


def bias_metrics(rows: list[dict], biases: list[float]) -> dict[str, float]:
    """행마다 기준 biases를 적용했을 때의 지표. rows와 biases는 같은 순서다

    ppp_accuracy               PPP 중 Yes로 답한 비율
    ppp_accuracy_change        그 비율의 변화 (기준 적용 후 - 원래)
    false_yes_rate             N이 1장인 케이스 전체 중 Yes로 답한 비율
    balanced_accuracy          ppp_accuracy와 (1 - false_yes_rate)의 평균
    n_ppp_yes, n_damaged       원래 Yes로 맞힌 PPP의 수, 그중 기준 적용 후 No가 된 수
    ppp_damage_rate            n_damaged / n_ppp_yes. 원래 맞힌 PPP가 없으면 nan
    n_false_yes, n_corrected   원래 False-Yes였던 케이스의 수, 그중 기준 적용 후 No가 된 수
    false_yes_correction_rate  n_corrected / n_false_yes. 원래 False-Yes가 없으면 nan
    "원래"는 b = 0, 즉 margin > 0이다.
    """
    ppp = [(row["logit_margin"], bias) for row, bias in zip(rows, biases) if row["variant"] == "PPP"]
    one_n = [(row["logit_margin"], bias) for row, bias in zip(rows, biases) if row["variant"] != "PPP"]
    ppp_accuracy = _rate(sum(margin > bias for margin, bias in ppp), len(ppp))
    false_yes_rate = _rate(sum(margin > bias for margin, bias in one_n), len(one_n))
    n_ppp_yes = sum(margin > 0 for margin, _ in ppp)
    n_damaged = sum(margin > 0 and not margin > bias for margin, bias in ppp)
    n_false_yes = sum(margin > 0 for margin, _ in one_n)
    n_corrected = sum(margin > 0 and not margin > bias for margin, bias in one_n)
    return {
        "ppp_accuracy": ppp_accuracy,
        "ppp_accuracy_change": ppp_accuracy - _rate(n_ppp_yes, len(ppp)),
        "false_yes_rate": false_yes_rate,
        "balanced_accuracy": (ppp_accuracy + 1 - false_yes_rate) / 2,
        "n_ppp_yes": n_ppp_yes,
        "n_damaged": n_damaged,
        "ppp_damage_rate": _rate(n_damaged, n_ppp_yes),
        "n_false_yes": n_false_yes,
        "n_corrected": n_corrected,
        "false_yes_correction_rate": _rate(n_corrected, n_false_yes),
    }


def sweep(rows: list[dict], grid: list[float]) -> dict[str, torch.Tensor]:
    """grid의 기준마다(모든 행에 같은 기준을 적용) 균형 정확도와 PPP 손상률: 각각 shape (기준 수,)

    bias_metrics를 기준마다 부른 것과 같은 값을 한 번에 계산한다. 원래 맞힌 PPP가 없으면 손상률은 nan이다.
    """
    margins = torch.tensor([row["logit_margin"] for row in rows], dtype=torch.float64)
    is_ppp = torch.tensor([row["variant"] == "PPP" for row in rows])
    yes = margins > torch.tensor(grid, dtype=torch.float64).unsqueeze(1)  # (기준 수, 행 수)
    ppp_accuracy = yes[:, is_ppp].double().mean(dim=1)
    false_yes_rate = yes[:, ~is_ppp].double().mean(dim=1)
    # 손상률은 "No가 된 수 / 원래 맞힌 수"로 바로 센다. 1에서 빼서 구하면 한도와 정확히 같은 값이 반올림으로 어긋날 수 있다
    return {
        "balanced_accuracy": (ppp_accuracy + 1 - false_yes_rate) / 2,
        "ppp_damage_rate": (~yes[:, is_ppp & (margins > 0)]).double().mean(dim=1),
    }


def best_bias(rows: list[dict], candidates: list[float]) -> float:
    """후보 중 rows에서 균형 정확도가 가장 높은 기준. 여럿이면 0에 가장 가까운 것, 그래도 여럿이면 양수인 것을 고른다"""
    balanced = sweep(rows, candidates)["balanced_accuracy"].tolist()
    highest = max(balanced)
    return min((bias for bias, value in zip(candidates, balanced) if value == highest),
               key=lambda bias: (abs(bias), -bias))


def largest_bias_within_damage(rows: list[dict], candidates: list[float], limit: float) -> float:
    """후보 중 rows에서 PPP 손상률이 limit 이하인 가장 큰 기준

    candidates는 0 이상이고 0을 포함해야 한다 (no_shift_candidates). b = 0의 손상률이 0이라 답이 항상 있다.
    기준이 클수록 손상률도 교정률도 줄지 않으므로, 한도 안에서 가장 큰 기준이 False-Yes를 가장 많이 고친다.
    """
    damage = sweep(rows, candidates)["ppp_damage_rate"].tolist()
    return max(bias for bias, value in zip(candidates, damage) if value <= limit)


def assign_biases(rows: list[dict], unit, pool, choose, hold_out: bool) -> list[float]:
    """행마다 적용할 기준을 정한다: rows와 같은 순서의 목록

    unit(row): 평가 단위. 같은 단위의 행은 같은 기준을 받는다
    pool(row): 기준을 고를 때 쓸 수 있는 행의 범위. pool이 같은 행들 안에서만 고른다
    choose(행들): 그 행들을 보고 기준 하나를 돌려주는 함수. 예: lambda rows: best_bias(rows, no_shift_candidates(rows))
    hold_out: True면 평가 단위 자신의 행을 빼고 고른다. 그 단위의 정답이 그 단위의 기준에 쓰이지 않는다
    자신을 뺐더니 고를 행이 남지 않으면 에러를 낸다 (예: 그룹이 하나뿐인 클래스의 class_cv).
    """
    bias_of_unit = {}
    for row in rows:
        key = unit(row)
        if key not in bias_of_unit:
            training = [other for other in rows if pool(other) == pool(row) and not (hold_out and unit(other) == key)]
            if not training:
                raise ValueError(f"평가 단위 {key}의 기준을 고를 행이 없음: 같은 범위에 다른 단위가 없다")
            bias_of_unit[key] = choose(training)
    return [bias_of_unit[unit(row)] for row in rows]


def selection_rules(rows: list[dict], fold_of: dict, candidates) -> dict[str, list[float]]:
    """균형 정확도가 가장 높은 기준을 고르는 네 규칙의, 행마다의 기준: {규칙 이름: [기준, ...]}

    fold_of: {클래스: 묶음 이름}. global은 그 묶음을 통째로 빼고 고른다 (클래스를 하나씩 빼려면 {클래스: 클래스})
    candidates(행들): 후보 b의 목록을 만드는 함수 (no_shift_candidates 또는 two_way_candidates).
        기준을 고르는 쪽의 행들로만 만든다
    original은 기준 0(모델의 원래 답)이다. 나머지 셋은 파일 맨 위의 설명과 같다.
    """
    def choose(training: list[dict]) -> float:
        return best_bias(training, candidates(training))

    def same_class(row: dict) -> str:
        return row["object"]

    return {
        "original": [0.0] * len(rows),
        "global": assign_biases(rows, lambda row: fold_of[row["object"]], lambda row: None, choose, hold_out=True),
        "class_cv": assign_biases(rows, lambda row: (row["object"], row["group_id"]), same_class, choose,
                                  hold_out=True),
        "class_oracle": assign_biases(rows, same_class, same_class, choose, hold_out=False),
    }


def damage_limited_biases(rows: list[dict], fold_of: dict, limit: float) -> list[float]:
    """PPP 손상률 한도의 기준선: 행마다, 그 클래스의 묶음을 뺀 나머지에서 손상률이 limit 이하인 가장 큰 기준

    기준과 후보는 학습 쪽에서만 만든다. 평가 쪽의 실제 손상률은 한도를 넘을 수 있고, 넘었는지는 평가 결과에서 그대로 본다.
    """
    def choose(training: list[dict]) -> float:
        return largest_bias_within_damage(training, no_shift_candidates(training), limit)

    return assign_biases(rows, lambda row: fold_of[row["object"]], lambda row: None, choose, hold_out=True)


def class_metrics(rows: list[dict], biases: list[float]) -> dict[str, dict[str, float]]:
    """클래스마다의 bias_metrics: {클래스: 지표} (클래스가 처음 나온 순서)"""
    by_class = {}
    for row, bias in zip(rows, biases):
        by_class.setdefault(row["object"], []).append((row, bias))
    return {name: bias_metrics([row for row, _ in chosen], [bias for _, bias in chosen])
            for name, chosen in by_class.items()}


def tradeoff_curve(rows: list[dict]) -> list[dict]:
    """기준을 0부터 올려 가며 본 맞바꿈 곡선: 기준마다 {"bias", ...bias_metrics의 지표}

    기준은 no_shift_candidates(rows), 즉 0과 행들에 실제로 있는 양수 margin 값 전부다. 그 사이의 값은 결과를 바꾸지
    않으므로 곡선이 정확하다. 고르는 과정이 없는 기술 통계이고, 같은 데이터로 그린 것이라 성능의 추정이 아니다.
    """
    return [{"bias": bias, **bias_metrics(rows, [bias] * len(rows))} for bias in no_shift_candidates(rows)]
