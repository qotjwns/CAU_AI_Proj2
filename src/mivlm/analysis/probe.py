"""
linear probe: 답변 위치의 벡터(pre_answer_hidden)에서 케이스의 속성을 선형 분류기로 맞힌다

모델은 건드리지 않는다. 저장된 벡터로 작은 분류기를 따로 학습하고, 학습에 쓰지 않은 케이스에서 맞히는지를 본다.
잘 맞히면 그 층의 벡터에 그 정보가 선형으로 읽히는 형태로 들어 있다는 뜻이다.
못 맞힌다고 정보가 없다는 뜻은 아니고(선형이 아닌 형태일 수 있다), 모델이 그 정보를 답에 쓰는지도 probe로는 알 수 없다.

분류기: softmax 회귀. 벡터 h를 학습용 케이스의 평균과 표준편차로 정규화한 z에 대해 점수 = W z + b,
    loss = cross entropy의 평균 + L2 x (W의 제곱합)
평가: 클래스(객체 종류) 단위 교차 검증. 같은 이미지가 학습과 평가 양쪽에 들어가지 않게 한다.

재검증 (STEP 9, held_out_probe_scores, nested_combined_scores, within_class_auroc)
    클래스가 달라도 같은 COCO 이미지를 쓸 수 있으므로, 묶음을 나눈 뒤 이미지로 한 번 더 거른다 (image_disjoint_training).
    margin에 probe를 더한 결합 점수는 2변수 로지스틱 회귀(fit_combiner)로 만든다. 결합 계수를 정할 때 쓰는 probe 점수는
    학습 클래스 안에서 다시 나눈 안쪽 교차 검증의 점수(학습에 쓰지 않은 점수)다. 그래서 결합 계수에 누수가 없다.
"""
from pathlib import Path

import torch

from mivlm.data.cases import image_linked_classes
from mivlm.evaluation.auroc import auroc
from mivlm.run_files import read_results

# 정규화 세기. 결과를 보고 고르지 않도록 하나로 고정한다. 벡터의 차원이 학습 케이스 수보다 많아서 0으로 두지 않는다
L2 = 0.01
N_FOLDS = 5
INNER_FOLDS = 4  # 결합 모형을 학습할 때 학습 클래스 안에서 다시 나누는 묶음 수
COMBINER_L2 = 1e-3  # 결합 회귀의 정규화 세기. 계수 두 개에만 건다 (절편은 빼고). 결과를 보기 전에 고정했다
PRESENCE_VARIANTS = ["PPP", "PPN", "PNP", "NPP"]  # N이 없거나 하나인 변형
ONE_N_VARIANTS = ["PPN", "PNP", "NPP"]  # N이 하나인 변형


def load_pre_answer(run_dir: Path, cases: list[dict]) -> tuple[list[dict], torch.Tensor]:
    """hidden state 실행 폴더의 pair 기록에서 케이스마다의 답변 위치 벡터를 모은다

    cases: load_cases()의 결과. 케이스의 클래스(object)와 변형(variant)을 case_id로 찾아 붙인다
    반환: (rows, hidden). 둘은 같은 순서다 (저장된 순서)
        rows: 케이스마다 {"case_id", "object", "variant", "image_ids", "logit_margin"}
        hidden: shape (케이스 수, 층 수 + 1, 벡터 크기), float32
    """
    cases_by_id = {case["case_id"]: case for case in cases}
    rows, hidden = [], []
    for result in read_results(run_dir):
        record = torch.load(run_dir / result["file"], map_location="cpu", weights_only=True)
        for saved in record["cases"].values():
            case = cases_by_id[saved["case_id"]]
            rows.append({"case_id": saved["case_id"], "object": case["object"], "variant": case["variant"],
                         "image_ids": case["image_ids"], "logit_margin": saved["logit_margin"]})
            hidden.append(saved["pre_answer_hidden"])
    return rows, torch.stack(hidden)


def class_folds(cases: list[dict], n_folds: int = N_FOLDS) -> dict[str, int]:
    """클래스를 n_folds개의 묶음으로 나눈다: {클래스 이름: 묶음 번호(0부터)}

    같은 이미지가 두 묶음에 들어가지 않게 한다. 이미지를 같이 쓰는 클래스들은 한 덩어리로 합쳐서 같은 묶음에 넣는다.
    덩어리를 큰 것부터 차례로, 그때 클래스 수가 가장 적은 묶음에 넣는다.
    무작위가 없고 클래스가 처음 나온 순서만 따르므로, 같은 케이스 파일이면 결과가 항상 같다.
    """
    sizes = [0] * n_folds
    folds = {}
    for group in sorted(image_linked_classes(cases), key=len, reverse=True):  # 크기가 같으면 처음 나온 순서를 유지한다
        fold = sizes.index(min(sizes))
        sizes[fold] += len(group)
        for name in group:
            folds[name] = fold
    return folds


def presence_labels(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """과제 "N이 있는가": N이 없는 PPP와 N이 하나인 변형을 구분한다

    반환: (쓸 케이스의 rows에서의 순번, 레이블). 레이블은 PPP가 0, PPN·PNP·NPP가 1
    """
    index = [i for i, row in enumerate(rows) if row["variant"] in PRESENCE_VARIANTS]
    return torch.tensor(index), torch.tensor([0 if rows[i]["variant"] == "PPP" else 1 for i in index])


def slot_labels(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """과제 "N이 어느 자리인가": N이 하나인 변형에서 그 자리를 맞힌다

    반환: (쓸 케이스의 rows에서의 순번, 레이블). 레이블은 N의 자리. NPP가 0, PNP가 1, PPN이 2
    """
    index = [i for i, row in enumerate(rows) if row["variant"] in ONE_N_VARIANTS]
    return torch.tensor(index), torch.tensor([rows[i]["variant"].index("N") for i in index])


def fit_probe(x: torch.Tensor, y: torch.Tensor, n_classes: int) -> dict:
    """softmax 회귀를 학습한다

    x: shape (케이스 수, 벡터 크기). y: shape (케이스 수,), 값은 0 ~ n_classes - 1
    반환: {"mean", "std": 정규화에 쓴 값, shape (벡터 크기,) / "weight": (n_classes, 벡터 크기) / "bias": (n_classes,)}
    모든 케이스에서 값이 같은 차원은 std를 무한대로 둬서 정규화한 값이 0이 되게 한다 (층 0은 모든 차원이 그렇다).
    """
    mean, std = x.mean(dim=0), x.std(dim=0)
    std[std < 1e-6] = float("inf")
    z = (x - mean) / std

    weight = torch.zeros(n_classes, x.shape[1], requires_grad=True)
    bias = torch.zeros(n_classes, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=500, line_search_fn="strong_wolfe")

    def loss_and_gradient():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(z @ weight.T + bias, y) + L2 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(loss_and_gradient)  # LBFGS는 이 한 번의 호출 안에서 수렴할 때까지(최대 500번) 반복한다
    return {"mean": mean, "std": std, "weight": weight.detach(), "bias": bias.detach()}


def probe_scores(probe: dict, x: torch.Tensor) -> torch.Tensor:
    """학습한 probe가 x에 매기는 클래스별 점수: shape (케이스 수, n_classes). 점수가 가장 높은 클래스가 예측이다"""
    z = (x - probe["mean"]) / probe["std"]
    return z @ probe["weight"].T + probe["bias"]


def cross_validated_scores(x: torch.Tensor, y: torch.Tensor, fold_of_case: torch.Tensor,
                           n_classes: int) -> torch.Tensor:
    """케이스마다, 그 케이스가 속한 묶음을 빼고 학습한 probe의 점수를 돌려준다: shape (케이스 수, n_classes)

    fold_of_case: 케이스마다의 묶음 번호, shape (케이스 수,)
    묶음 하나를 평가용으로 두고 나머지로 학습하기를 묶음 수만큼 반복한다. 모든 케이스가 평가용이 한 번씩 된다.
    """
    scores = torch.empty(len(y), n_classes)
    for fold in fold_of_case.unique():
        test = fold_of_case == fold
        probe = fit_probe(x[~test], y[~test], n_classes)
        scores[test] = probe_scores(probe, x[test])
    return scores


def accuracy_by_layer(predictions: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask가 고른 케이스에서 층마다의 정확도: shape (층 수 + 1,)

    predictions: 층마다의 예측, shape (층 수 + 1, 케이스 수). labels: shape (케이스 수,). mask: 쓸 케이스, bool, shape (케이스 수,)
    """
    return (predictions[:, mask] == labels[mask]).float().mean(dim=1)



# ------------------------------------------------------------------ 재검증 (STEP 9)


def n_present_logit(scores: torch.Tensor) -> torch.Tensor:
    """presence probe의 점수 (케이스 수, 2)에서 "N 있음"의 원시 logit: 점수[:, 1] - 점수[:, 0], float64

    확률이 아니라 logit을 쓴다. 클래스가 두 개인 softmax에서 이 값이 "N 있음" 쪽 log-odds다.
    """
    return (scores[:, 1] - scores[:, 0]).double()


def image_disjoint_training(train: list[int], test: list[int], rows: list[dict]) -> tuple[list[int], int]:
    """학습 쪽 케이스 중 평가 쪽 케이스와 이미지를 하나라도 같이 쓰는 케이스를 뺀다: (남긴 순번, 뺀 수)

    train, test: rows에서의 순번. rows[i]["image_ids"]를 본다. 순서를 유지하고 무작위가 없다.
    클래스로 묶음을 나눠도 같은 COCO 이미지가 다른 클래스의 케이스에 들어 있을 수 있어서, 이미지로 한 번 더 거른다.
    """
    test_images = {image_id for i in test for image_id in rows[i]["image_ids"]}
    kept = [i for i in train if not test_images.intersection(rows[i]["image_ids"])]
    return kept, len(train) - len(kept)


def held_out_probe_scores(x: torch.Tensor, labels: torch.Tensor, rows: list[dict],
                          fold_of_case: list[int]) -> tuple[torch.Tensor, list[int]]:
    """presence probe의 묶음 밖 점수: 케이스마다 그 묶음을 빼고(이미지가 겹치는 학습 케이스도 빼고) 학습한 probe의
    "N 있음" logit. 반환: (점수 (케이스 수,) float64, 묶음마다 이미지 때문에 뺀 학습 케이스 수)

    x: (케이스 수, 벡터 크기). labels: 0(PPP) 또는 1(N 하나). rows, fold_of_case: x와 같은 순서
    """
    scores = torch.empty(len(rows), dtype=torch.float64)
    dropped = []
    for fold in sorted(set(fold_of_case)):
        test = [i for i, k in enumerate(fold_of_case) if k == fold]
        train, n_dropped = image_disjoint_training([i for i, k in enumerate(fold_of_case) if k != fold], test, rows)
        probe = fit_probe(x[train], labels[train], 2)
        scores[test] = n_present_logit(probe_scores(probe, x[test]))
        dropped.append(n_dropped)
    return scores, dropped


def fit_combiner(features: torch.Tensor, labels: torch.Tensor) -> dict:
    """결합 모형: 특징 2개로 "N 있음"을 맞히는 로지스틱 회귀

    features: (케이스 수, 2) = [-margin, probe의 "N 있음" logit]. labels: 0 또는 1
    특징은 이 학습 케이스들의 평균과 표준편차로 표준화하고, 같은 값을 평가에도 그대로 쓴다 (combiner_scores).
    loss = binary cross entropy의 평균 + COMBINER_L2 x (계수의 제곱합). 최적화는 LBFGS(strong Wolfe), 최대 500번 반복,
    torch의 기본 수렴 기준이다. 반환: {"mean", "std": (2,), "weight": (2,), "bias": 스칼라}, float64
    """
    features = features.double()
    mean, std = features.mean(dim=0), features.std(dim=0)
    std[std < 1e-12] = float("inf")  # 값이 모두 같은 특징은 0이 되게 한다
    z = (features - mean) / std
    weight = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=500, line_search_fn="strong_wolfe")

    def loss_and_gradient():
        optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(z @ weight + bias, labels.double())
                + COMBINER_L2 * weight.square().sum())
        loss.backward()
        return loss

    optimizer.step(loss_and_gradient)
    return {"mean": mean, "std": std, "weight": weight.detach(), "bias": bias.detach()}


def combiner_scores(combiner: dict, features: torch.Tensor) -> torch.Tensor:
    """학습한 결합 모형의 "N 있음" 점수(log-odds): (케이스 수,). 표준화에는 학습 때의 평균과 표준편차를 쓴다"""
    z = (features.double() - combiner["mean"]) / combiner["std"]
    return z @ combiner["weight"] + combiner["bias"]


def _distribution(values: torch.Tensor) -> dict[str, float]:
    """점수 분포의 요약: 평균, 표준편차"""
    return {"mean": values.mean().item(), "std": values.std().item()}


def nested_combined_scores(x: torch.Tensor, margins: torch.Tensor, labels: torch.Tensor, rows: list[dict],
                           fold_of_case: list[int]) -> dict:
    """margin과 probe를 합친 결합 점수를 묶음 밖에서 매긴다 (중첩 교차 검증)

    바깥 묶음마다
        1. 학습 = 다른 묶음의 케이스에서 평가 묶음과 이미지가 겹치는 케이스를 뺀 것
        2. 학습 클래스를 INNER_FOLDS개로 다시 나눠(class_folds, 이미지로 한 번 더 거름) 학습 케이스마다
           학습에 쓰지 않은 probe 점수(안쪽 점수)를 만든다
        3. [-margin, 안쪽 점수]로 결합 계수를 정한다 (fit_combiner)
        4. 학습 전체로 probe를 다시 학습해 평가 묶음에 점수(바깥 점수)를 매기고, 결합 계수로 결합 점수를 낸다
    x: (케이스 수, 벡터 크기). margins: (케이스 수,). labels: 0 또는 1. rows: image_ids와 object가 있는 행
    반환: {"probe": 바깥 점수, "combined": 결합 점수 (둘 다 (케이스 수,) float64),
           "folds": 묶음마다 {"fold", "n_test", "n_train", "dropped_outer", "dropped_inner",
                             "inner_scores", "outer_scores" (점수 분포의 평균과 표준편차), "combiner"}}
    안쪽 점수는 바깥 점수보다 적은 클래스로 학습한 probe에서 나온다. 두 분포를 같이 적어 두고 결과를 해석할 때 본다.
    """
    margins = margins.double()
    probe_out = torch.empty(len(rows), dtype=torch.float64)
    combined = torch.empty(len(rows), dtype=torch.float64)
    folds = []
    for fold in sorted(set(fold_of_case)):
        test = [i for i, k in enumerate(fold_of_case) if k == fold]
        train, dropped_outer = image_disjoint_training([i for i, k in enumerate(fold_of_case) if k != fold], test, rows)

        inner_fold_of = class_folds([rows[i] for i in train], INNER_FOLDS)
        inner = [inner_fold_of[rows[i]["object"]] for i in train]
        inner_scores, dropped_inner = held_out_probe_scores(x[train], labels[train], [rows[i] for i in train], inner)
        combiner = fit_combiner(torch.stack([-margins[train], inner_scores], dim=1), labels[train])

        probe = fit_probe(x[train], labels[train], 2)
        outer_scores = n_present_logit(probe_scores(probe, x[test]))
        probe_out[test] = outer_scores
        combined[test] = combiner_scores(combiner, torch.stack([-margins[test], outer_scores], dim=1))
        folds.append({
            "fold": fold, "n_test": len(test), "n_train": len(train),
            "dropped_outer": dropped_outer, "dropped_inner": sum(dropped_inner),
            "inner_scores": _distribution(inner_scores), "outer_scores": _distribution(outer_scores),
            "combiner": {key: value.tolist() for key, value in combiner.items()},
        })
    return {"probe": probe_out, "combined": combined, "folds": folds}


def within_class_auroc(scores: torch.Tensor, labels: torch.Tensor, objects: list[str]) -> dict[str, float]:
    """클래스마다 그 클래스의 케이스만으로 낸 AUROC: {클래스: 값} (클래스가 처음 나온 순서)

    레이블 1(N 있음)의 점수가 더 높을 확률이다. 클래스마다 점수에 같은 값을 더해도 바뀌지 않으므로 클래스 성향을 뺀다.
    그 클래스에 한쪽 레이블이 없으면 nan이다.
    """
    result = {}
    for name in dict.fromkeys(objects):
        mask = torch.tensor([value == name for value in objects])
        result[name] = auroc(scores[mask], labels[mask]).item()
    return result
