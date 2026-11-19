"""
linear probe: 답변 위치의 벡터(pre_answer_hidden)에서 케이스의 속성을 선형 분류기로 맞힌다

모델은 건드리지 않는다. 저장된 벡터로 작은 분류기를 따로 학습하고, 학습에 쓰지 않은 케이스에서 맞히는지를 본다.
잘 맞히면 그 층의 벡터에 그 정보가 선형으로 읽히는 형태로 들어 있다는 뜻이다.
못 맞힌다고 정보가 없다는 뜻은 아니고(선형이 아닌 형태일 수 있다), 모델이 그 정보를 답에 쓰는지도 probe로는 알 수 없다.

분류기: softmax 회귀. 벡터 h를 학습용 케이스의 평균과 표준편차로 정규화한 z에 대해 점수 = W z + b,
    loss = cross entropy의 평균 + L2 x (W의 제곱합)
평가: 클래스(객체 종류) 단위 교차 검증. 같은 이미지가 학습과 평가 양쪽에 들어가지 않게 한다.
"""
from pathlib import Path

import torch

from mivlm.data.cases import image_linked_classes
from mivlm.run_files import read_results

# 정규화 세기. 결과를 보고 고르지 않도록 하나로 고정한다. 벡터의 차원이 학습 케이스 수보다 많아서 0으로 두지 않는다
L2 = 0.01
N_FOLDS = 5
PRESENCE_VARIANTS = ["PPP", "PPN", "PNP", "NPP"]  # N이 없거나 하나인 변형
ONE_N_VARIANTS = ["PPN", "PNP", "NPP"]  # N이 하나인 변형


def load_pre_answer(run_dir: Path, cases: list[dict]) -> tuple[list[dict], torch.Tensor]:
    """hidden state 실행 폴더의 pair 기록에서 케이스마다의 답변 위치 벡터를 모은다

    cases: load_cases()의 결과. 케이스의 클래스(object)와 변형(variant)을 case_id로 찾아 붙인다
    반환: (rows, hidden). 둘은 같은 순서다 (저장된 순서)
        rows: 케이스마다 {"case_id", "object", "variant", "logit_margin"}
        hidden: shape (케이스 수, 층 수 + 1, 벡터 크기), float32
    """
    cases_by_id = {case["case_id"]: case for case in cases}
    rows, hidden = [], []
    for result in read_results(run_dir):
        record = torch.load(run_dir / result["file"], map_location="cpu", weights_only=True)
        for saved in record["cases"].values():
            case = cases_by_id[saved["case_id"]]
            rows.append({"case_id": saved["case_id"], "object": case["object"], "variant": case["variant"],
                         "logit_margin": saved["logit_margin"]})
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

