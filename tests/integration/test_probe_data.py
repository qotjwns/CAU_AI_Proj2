"""
probe 재검증(STEP 9)이 쓰는 실제 hidden state 기록 검사 (모델은 쓰지 않고, 저장된 3B float32 기록만 읽는다)
pytest tests/integration/test_probe_data.py -v

1. margin과 마지막 벡터의 대응: hidden state 실행은 margin(첫 생성 위치의 가공 전 logit 차이)과 pre_answer_hidden을
   같은 forward에서 저장했다. 마지막 항목은 최종 norm 뒤의 벡터이고 Qwen2.5-VL의 LM head에는 bias가 없으므로
   m = (w_Yes - w_No)ᵀ h_last 이어야 한다.
   검사: 케이스를 고정 분할한다 (순번을 10으로 나눈 나머지가 9인 240개가 평가, 나머지 2,160개가 학습).
   학습 케이스로 [h_last, 1]의 최소제곱 계수를 구하고(학습 행 2,160 > 미지수 2,049), 쓰지 않은 평가 케이스에서
   잔차를 잰다. 같은 케이스로 맞추고 재면 차원이 커서 선형이 아닌 관계도 잔차가 작아질 수 있으므로 평가를 따로 둔다.
   이 검사는 관계가 선형식과 맞는지를 볼 뿐, 구한 계수가 실제 LM head의 w_Yes - w_No와 같은지는 보지 않는다.
   LM head 가중치를 직접 쓰는 검사는 가중치가 이 기기의 캐시에 없어서 대신 이 방법을 쓴다.
2. 바깥 클래스 묶음끼리 이미지가 겹치지 않는다.
기록이 없으면 건너뛴다. 잔차와 조건수는 pytest -s로 볼 수 있게 출력한다.
"""
import pytest
import torch

from mivlm.analysis.probe import N_FOLDS, class_folds, image_disjoint_training, load_pre_answer
from mivlm.config import PREDICTIONS_DIR
from mivlm.data.cases import load_cases

RUN_DIR = PREDICTIONS_DIR / "03_hidden_states" / "qwen2.5-vl-3b_float32"

pytestmark = pytest.mark.skipif(not (RUN_DIR / "results.jsonl").exists(),
                                reason="3B float32 hidden state 기록이 없음")


@pytest.fixture(scope="module")
def loaded():
    return load_pre_answer(RUN_DIR, load_cases())


HELD_OUT_EVERY = 10  # 순번을 이 수로 나눈 나머지가 HELD_OUT_EVERY - 1인 케이스가 평가용이다


def _held_out_fit(hidden_layer: torch.Tensor, margins: torch.Tensor) -> dict[str, float]:
    """학습 케이스로 margin ~ [벡터, 1]의 최소제곱 계수를 구하고, 평가 케이스에서 잰 값들

    held_out_relative_residual: 평가 케이스 잔차의 RMS / 평가 케이스 margin의 표준편차
    held_out_max_residual: 평가 케이스 잔차 절댓값의 최댓값 (logit 단위)
    condition_number: 학습 설계 행렬 [벡터, 1]의 조건수
    """
    held_out = torch.arange(len(margins)) % HELD_OUT_EVERY == HELD_OUT_EVERY - 1
    design = torch.cat([hidden_layer.double(), torch.ones(len(margins), 1, dtype=torch.float64)], dim=1)
    solution = torch.linalg.lstsq(design[~held_out], margins[~held_out]).solution
    residual = margins[held_out] - design[held_out] @ solution
    return {
        "n_train": int((~held_out).sum()), "n_unknowns": design.shape[1], "n_held_out": int(held_out.sum()),
        "held_out_relative_residual": (residual.square().mean().sqrt() / margins[held_out].std()).item(),
        "held_out_max_residual": residual.abs().max().item(),
        "condition_number": torch.linalg.cond(design[~held_out]).item(),
    }


def test_the_margin_is_a_linear_function_of_the_last_vector(loaded):
    rows, hidden = loaded
    margins = torch.tensor([row["logit_margin"] for row in rows], dtype=torch.float64)

    last = _held_out_fit(hidden[:, -1], margins)
    middle = _held_out_fit(hidden[:, hidden.shape[1] // 2], margins)  # 보조 진단. 통과 기준에 쓰지 않는다
    print(f"\n마지막 항목: {last}\n가운데 층 {hidden.shape[1] // 2} (보조 진단): {middle}")

    assert last["n_train"] > last["n_unknowns"]  # 학습 행이 미지수보다 많아야 계수가 하나로 정해진다
    assert last["held_out_relative_residual"] < 1e-3  # 쓰지 않은 케이스에서도 float32 연산 오차 수준


def test_the_outer_class_folds_share_no_image(loaded):
    rows, _ = loaded
    folds = class_folds(load_cases())
    fold_of_case = [folds[row["object"]] for row in rows]

    for fold in range(N_FOLDS):
        train = [i for i, k in enumerate(fold_of_case) if k != fold]
        test = [i for i, k in enumerate(fold_of_case) if k == fold]
        assert image_disjoint_training(train, test, rows)[1] == 0
