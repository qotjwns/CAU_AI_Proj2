"""
linear probe 실행: 답변 위치의 벡터에서 "N이 있는가", "N이 어느 자리인가"를 층마다 맞혀 본다
python experiments/04_probe/run.py qwen2.5-vl-3b             # hidden state 실행 폴더 이름
python experiments/04_probe/run.py qwen2.5-vl-3b_float32

입력: outputs/predictions/03_hidden_states/<실행 이름>/의 pair 기록 (케이스마다의 pre_answer_hidden)
출력: outputs/predictions/04_probe/<실행 이름>/probe.pt (과제마다, 층마다, 케이스별로 probe가 매긴 클래스별 점수)
모델은 쓰지 않고 CPU로 돈다. 결과 표는 summarize.py로 본다.

과제 두 가지 (mivlm.analysis.probe 참고)
    presence: PPP와 N이 하나인 변형(PPN, PNP, NPP)을 구분한다. 모델이 풀어야 하는 문제와 같다
    slot: N이 하나인 변형에서 N이 자리 1, 2, 3 중 어디인지 맞힌다
과제마다 레이블을 무작위로 섞어 학습한 probe(control)도 같이 돌린다. probe가 우연히 맞히는 수준을 보기 위한 것이다.
"""
import argparse

import torch
from tqdm import tqdm

from mivlm.analysis.probe import (
    L2, N_FOLDS, class_folds, cross_validated_scores, load_pre_answer, presence_labels, slot_labels,
)
from mivlm.config import PREDICTIONS_DIR
from mivlm.data.cases import load_run_cases
from mivlm.runner import git_state

EXPERIMENT = "04_probe"
SOURCE_EXPERIMENT = "03_hidden_states"
CONTROL_SEED = 0  # control에서 레이블을 섞는 순서를 정하는 seed


def main() -> None:
    parser = argparse.ArgumentParser(description="답변 위치 벡터의 linear probe")
    parser.add_argument("run", help="outputs/predictions/03_hidden_states/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    source_dir = PREDICTIONS_DIR / SOURCE_EXPERIMENT / args.run
    cases = load_run_cases(source_dir)  # hidden state 실행이 쓴 풀의 케이스 (해시 확인)
    rows, hidden = load_pre_answer(source_dir, cases)
    folds = class_folds(cases)
    fold_of_case = torch.tensor([folds[row["object"]] for row in rows])
    print(f"{args.run}: cases {len(rows)}, hidden {tuple(hidden.shape)}, classes per fold "
          f"{[sum(fold == k for fold in folds.values()) for k in range(N_FOLDS)]}")

    tasks = {}
    for name, (index, labels) in {"presence": presence_labels(rows), "slot": slot_labels(rows)}.items():
        n_classes = labels.max().item() + 1
        shuffled = labels[torch.randperm(len(labels), generator=torch.Generator().manual_seed(CONTROL_SEED))]
        scores, control_scores = [], []
        for layer in tqdm(range(hidden.shape[1]), desc=name):
            x = hidden[index, layer]
            scores.append(cross_validated_scores(x, labels, fold_of_case[index], n_classes))
            control_scores.append(cross_validated_scores(x, shuffled, fold_of_case[index], n_classes))
        tasks[name] = {
            "case_ids": [rows[i]["case_id"] for i in index.tolist()],
            "variants": [rows[i]["variant"] for i in index.tolist()],
            "logit_margins": torch.tensor([rows[i]["logit_margin"] for i in index.tolist()]),
            "labels": labels,
            "scores": torch.stack(scores),  # (층 수 + 1, 케이스 수, 클래스 수). 예측은 점수가 가장 높은 클래스
            "control_scores": torch.stack(control_scores),
        }

    out_dir = PREDICTIONS_DIR / EXPERIMENT / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "source_run": args.run,
        "settings": {"l2": L2, "n_folds": N_FOLDS, "control_seed": CONTROL_SEED},
        "git": git_state(),
        "folds": folds,
        "tasks": tasks,
    }, out_dir / "probe.pt")
    print(f"[ok] {out_dir / 'probe.pt'}")


if __name__ == "__main__":
    main()
