"""
linear probe 결과 요약: 층마다의 probe 정확도를 표로 출력
python experiments/04_probe/summarize.py qwen2.5-vl-3b

run.py가 저장한 outputs/predictions/04_probe/<실행 이름>/probe.pt를 읽는다. 결과는 터미널에만 출력한다.
정확도는 모두 "학습에 쓰지 않은 케이스"에 대한 것이다 (클래스 단위 교차 검증).
모델의 답은 logit margin의 부호로 본다 (양수면 Yes).

presence 표의 열
    probe, control  전체 정확도. control은 레이블을 섞어 학습한 probe
    balanced        PPP의 정확도와 N 하나인 케이스의 정확도의 평균. 한쪽으로 찍으면 0.5다
    PPP, oneN       PPP 케이스의 정확도(N 없음을 맞힘), N 하나인 케이스의 정확도(N 있음을 맞힘)
    N_right, N_FY   N 하나인 케이스 중 모델이 맞힌 케이스, 모델이 False-Yes인 케이스의 정확도
    auroc           "N 있음" 점수가 PPP와 N 하나인 케이스를 가르는 정도. 기준점에 좌우되지 않는다
    auroc_yes       모델이 Yes라고 답한 케이스만으로 계산한 auroc (PPP 대 False-Yes)
AUROC는 다섯 묶음의 점수를 한데 모아 계산한다. 묶음마다 다른 probe가 매긴 점수라서 묶음별로 계산한 값과 조금 다를 수 있다.
margin 기준선은 probe 없이 모델의 logit margin만으로 계산한 AUROC다 (margin이 낮을수록 "N 있음"). probe가 이보다 높아야
"모델의 Yes/No 확신도에 이미 드러난 것 이상"을 읽었다고 볼 수 있다.

읽을 때 주의할 것
    probe가 잘 맞힌다 = 그 층의 벡터에 정보가 선형으로 읽히는 형태로 있다. 모델이 그 정보를 썼다는 뜻은 아니다
    probe가 못 맞힌다 = 정보가 없거나, 클래스마다 다른 방향으로 들어 있거나, 선형이 아니다. 셋을 구분할 수 없다
    층 0은 모든 케이스의 벡터가 같으므로 control과 같은 수준이어야 한다
"""
import argparse

import torch

from mivlm.analysis.probe import accuracy_by_layer
from mivlm.config import PREDICTIONS_DIR
from mivlm.evaluation.auroc import auroc
from mivlm.evaluation.report import print_layer_table

EXPERIMENT = "04_probe"


def main() -> None:
    parser = argparse.ArgumentParser(description="linear probe 결과 요약")
    parser.add_argument("run", help="outputs/predictions/04_probe/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    saved = torch.load(PREDICTIONS_DIR / EXPERIMENT / args.run / "probe.pt", map_location="cpu", weights_only=True)
    print(f"{args.run}: settings {saved['settings']}, git {saved['git']}")

    # 과제 1: N이 있는가 (레이블 0 = PPP, 1 = N이 하나)
    task = saved["tasks"]["presence"]
    labels, predictions = task["labels"], task["scores"].argmax(dim=-1)
    n_score = task["scores"][..., 1] - task["scores"][..., 0]  # "N 있음" 쪽 점수, shape (층 수 + 1, 케이스 수)
    margin_score = -task["logit_margins"][None]  # margin이 낮을수록 "N 있음" 쪽. 층이 하나인 것처럼 다룬다
    everything = torch.ones_like(labels, dtype=torch.bool)
    one_n = labels == 1
    model_yes = task["logit_margins"] > 0
    model_right = torch.where(one_n, ~model_yes, model_yes)  # PPP는 Yes가, N이 하나인 변형은 No가 정답
    print(f"\n[presence: N이 있는가] 케이스 {len(labels)}개 (PPP {(~one_n).sum().item()}, N 하나 {one_n.sum().item()})")
    print(f"  전부 'N 있음'으로 찍을 때의 정확도 {one_n.float().mean().item():.1%}, "
          f"같은 케이스에서 모델의 정확도 {model_right.float().mean().item():.1%}")
    print(f"  N 하나인 케이스 중 모델이 No로 맞힘 {(one_n & ~model_yes).sum().item()}개, "
          f"Yes로 틀림(False-Yes) {(one_n & model_yes).sum().item()}개")
    print(f"  margin 기준선(AUROC): 전체 {auroc(margin_score, labels).item():.3f}, 모델이 Yes인 케이스만 "
          f"{auroc(margin_score[:, model_yes], labels[model_yes]).item():.3f}")
    ppp_accuracy = accuracy_by_layer(predictions, labels, ~one_n)
    one_n_accuracy = accuracy_by_layer(predictions, labels, one_n)
    print_layer_table("presence: probe 정확도와 AUROC (열의 뜻은 파일 맨 위)", {
        "probe": accuracy_by_layer(predictions, labels, everything),
        "control": accuracy_by_layer(task["control_scores"].argmax(dim=-1), labels, everything),
        "balanced": (ppp_accuracy + one_n_accuracy) / 2,
        "PPP": ppp_accuracy,
        "oneN": one_n_accuracy,
        "N_right": accuracy_by_layer(predictions, labels, one_n & ~model_yes),
        "N_FY": accuracy_by_layer(predictions, labels, one_n & model_yes),
        "auroc": auroc(n_score, labels),
        "auroc_yes": auroc(n_score[:, model_yes], labels[model_yes]),
    }, ".3f")

    # 과제 2: N이 어느 자리인가 (레이블 0, 1, 2 = 자리 1, 2, 3). 케이스는 모두 N이 하나인 변형
    task = saved["tasks"]["slot"]
    labels, predictions = task["labels"], task["scores"].argmax(dim=-1)
    everything = torch.ones_like(labels, dtype=torch.bool)
    model_yes = task["logit_margins"] > 0
    print(f"\n[slot: N이 어느 자리인가] 케이스 {len(labels)}개, 찍을 때의 정확도 {1 / 3:.1%}")
    print(f"  모델이 No로 맞힘 {(~model_yes).sum().item()}개, Yes로 틀림(False-Yes) {model_yes.sum().item()}개")
    print("  열: probe = 전체, control = 레이블을 섞어 학습, right = 모델이 맞힌 케이스, FY = 모델이 False-Yes인 케이스,")
    print("      slot1, slot2, slot3 = N이 그 자리에 있는 케이스 (NPP, PNP, PPN)")
    print_layer_table("slot: probe 정확도", {
        "probe": accuracy_by_layer(predictions, labels, everything),
        "control": accuracy_by_layer(task["control_scores"].argmax(dim=-1), labels, everything),
        "right": accuracy_by_layer(predictions, labels, ~model_yes),
        "FY": accuracy_by_layer(predictions, labels, model_yes),
        "slot1": accuracy_by_layer(predictions, labels, labels == 0),
        "slot2": accuracy_by_layer(predictions, labels, labels == 1),
        "slot3": accuracy_by_layer(predictions, labels, labels == 2),
    }, ".3f")


if __name__ == "__main__":
    main()
