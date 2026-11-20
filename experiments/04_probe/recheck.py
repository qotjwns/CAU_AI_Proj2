"""
probe 재검증 (STEP 9): margin에 답변 위치 벡터의 probe를 더하면, 학습에 쓰지 않은 클래스 안에서 PPP와 N 하나를 더 잘 가르는가
python experiments/04_probe/recheck.py qwen2.5-vl-3b_float32    # 주 실행
python experiments/04_probe/recheck.py qwen2.5-vl-3b            # 보조 실행
python experiments/04_probe/recheck.py qwen2.5-vl-7b            # 보조 실행

입력: outputs/predictions/03_hidden_states/<실행 이름>/의 pair 기록 (기존 탐색 데이터 311장, 2,400 케이스)
출력: outputs/predictions/04_probe/<실행 이름>/recheck.pt. 기존 probe.pt는 덮어쓰지 않는다. 표는 터미널에 출력한다.
모델은 쓰지 않고 CPU로 돈다. 정의는 mivlm.analysis.probe 참고.

과제: presence. PPP(레이블 0)와 N이 하나인 변형 PPN, PNP, NPP(레이블 1)를 구분한다. 점수는 모두 "N 있음" 쪽이 크다
    margin 단독  -margin. 학습하지 않는다
    probe 단독   마지막 항목(최종 norm 뒤의 벡터)의 probe가 매긴 "N 있음" logit (바깥 묶음 밖 점수)
    결합         [-margin, probe 점수]의 로지스틱 회귀. 계수는 안쪽 교차 검증의 점수로 정한다 (nested_combined_scores)
평가: 클래스 단위 5묶음 교차 검증 (class_folds). 평가 묶음과 이미지가 겹치는 학습 케이스는 뺀다 (안쪽도 같다)
    지표는 클래스 내부 AUROC의 클래스 평균이다. 클래스마다 같은 무게를 갖는다

판정 기준 (실행 전에 정했다. 결과를 본 뒤에 바꾸지 않는다)
    주 지표: Δ = AUROC_wc(결합) - AUROC_wc(margin 단독), 주 실행의 마지막 항목. 클래스마다의 차이를 짝지어 평균한다
    구간: 클래스를 단위로 10,000번 다시 뽑은 95% 구간 (seed 고정)
    실질적 효과: 하한 > +0.02이면 실질적 추가 판별력 지지, 상한 < -0.02이면 실질적 성능 저하,
        구간 전체가 -0.02 ~ +0.02 안이면 실질적 차이 없음, 그 밖에는 실질적 효과 미결정.
        0.02는 이 실험에서 미리 정한 탐색적 효과 크기 기준이다 (Fixed Bias의 2%p와 같은 단위가 아니다)
    통계적 방향(구간이 0을 제외하는가)은 따로 적는다
    보조 지표 (같은 방식, 판정 문구를 같이 적되 주 판정이 아니다)
        모델이 Yes라고 답한 케이스만의 Δ (PPP 대 False-Yes). 그런 케이스가 양쪽에 다 있는 클래스만 쓴다
        probe 단독 - margin 단독 (기존 비교의 재현)
    모든 층의 probe 단독 AUROC와 레이블을 섞은 대조(control)는 판정 없이 보고한다
"""
import argparse
import math
import statistics

import torch
from tqdm import tqdm

from mivlm.analysis.class_sensitivity import direction_verdict, mean_interval
from mivlm.analysis.evidence_link import practical_verdict
from mivlm.analysis.probe import (
    COMBINER_L2, INNER_FOLDS, L2, N_FOLDS, class_folds, held_out_probe_scores, image_disjoint_training,
    load_pre_answer, nested_combined_scores, presence_labels, within_class_auroc,
)
from mivlm.config import PREDICTIONS_DIR
from mivlm.data.cases import load_cases
from mivlm.evaluation.report import interval_text
from mivlm.runner import git_state

EXPERIMENT = "04_probe"
SOURCE_EXPERIMENT = "03_hidden_states"
MAIN_RUN = "qwen2.5-vl-3b_float32"  # 주 판정을 내리는 실행
PRACTICAL_DELTA = 0.02  # 실질적 효과의 기준 (AUROC). 이 실험에서 미리 정한 탐색적 기준이다
CONTROL_SEED = 0  # control에서 레이블을 섞는 순서를 정하는 seed
DIRECTION = {"지지": "0보다 큼", "반증": "0보다 작음", "미결정": "방향 불명"}
PRACTICAL = {"앞 모델이 실질적으로 우월": "실질적 추가 판별력 지지",
             "뒤 모델이 실질적으로 우월": "실질적 성능 저하",
             "실질적 차이 없음": "실질적 차이 없음", "미결정": "실질적 효과 미결정"}
LIMITS = """[한계]
  기존 탐색 데이터(311장)로 한 재검증이라 탐색 결과다. 독립 확증에는 확증 풀의 hidden state가 필요하다.
  결과가 긍정적이어도 모델 안에서 부정 증거가 억제된다는 인과적 증거가 아니다. 마지막 벡터에 margin 말고도
  선형으로 읽히는 판별 정보가 있다는 근거까지다. 모델이 그 정보를 쓰는지는 probe로 알 수 없다.
  결합 계수는 안쪽 교차 검증의 점수(더 적은 클래스로 학습한 probe)로 정하고 바깥 점수에 적용한다.
  두 점수의 분포가 다르면 결합이 나빠질 수 있다. 묶음별 표의 분포를 같이 본다.
  클래스 내부 AUROC는 클래스마다 PPP 20개와 N 하나인 케이스 60개로 낸 값이라 흔들림이 크다.
  구간은 클래스(20개)를 단위로 다시 뽑아 만든다. 단위가 적어 실제보다 조금 좁을 수 있다."""


def mean_defined(values: dict[str, float]) -> float:
    """nan을 뺀 값들의 평균"""
    defined = [value for value in values.values() if not math.isnan(value)]
    return statistics.mean(defined) if defined else float("nan")


def paired_difference(first: dict[str, float], second: dict[str, float]) -> list[float]:
    """클래스마다 first - second. 어느 한쪽이라도 정할 수 없는 클래스는 뺀다"""
    return [first[name] - second[name] for name in first if not (math.isnan(first[name]) or math.isnan(second[name]))]


def verdict_line(label: str, differences: list[float]) -> tuple[str, dict]:
    """차이의 클래스 평균과 95% 구간, 통계적 방향, 실질적 효과를 한 줄로. 기록용 값도 돌려준다"""
    mean, low, high = mean_interval(differences)
    direction = DIRECTION[direction_verdict(low, high)]
    practical = PRACTICAL[practical_verdict(low, high, PRACTICAL_DELTA)]
    line = f"  {label:<34}{interval_text(mean, low, high):<28}{direction:<10}{practical} ({len(differences)}개 클래스)"
    return line, {"mean": mean, "low": low, "high": high, "n_classes": len(differences),
                  "direction": direction, "practical": practical}


def main() -> None:
    parser = argparse.ArgumentParser(description="probe 재검증 (STEP 9)")
    parser.add_argument("run", help="outputs/predictions/03_hidden_states/ 아래의 실행 폴더 이름")
    args = parser.parse_args()

    cases = load_cases()
    all_rows, hidden = load_pre_answer(PREDICTIONS_DIR / SOURCE_EXPERIMENT / args.run, cases)
    index, labels = presence_labels(all_rows)
    rows = [all_rows[i] for i in index.tolist()]
    margins = torch.tensor([row["logit_margin"] for row in rows], dtype=torch.float64)
    objects = [row["object"] for row in rows]
    folds = class_folds(cases)
    fold_of_case = [folds[row["object"]] for row in rows]
    last = hidden.shape[1] - 1

    # 바깥 묶음마다 이미지 때문에 빼야 하는 학습 케이스의 수. class_folds가 이미지로 이어진 클래스를 묶으므로 0이어야 한다
    outer_dropped = [image_disjoint_training([i for i, k in enumerate(fold_of_case) if k != fold],
                                             [i for i, k in enumerate(fold_of_case) if k == fold], rows)[1]
                     for fold in range(N_FOLDS)]

    nested = nested_combined_scores(hidden[index, last], margins, labels, rows, fold_of_case)
    model_yes = margins > 0
    scores = {"margin": -margins, "probe": nested["probe"], "combined": nested["combined"]}
    by_class = {name: within_class_auroc(score, labels, objects) for name, score in scores.items()}
    by_class_yes = {name: within_class_auroc(score[model_yes], labels[model_yes],
                                             [objects[i] for i in range(len(rows)) if model_yes[i]])
                    for name, score in scores.items()}

    print(f"{'=' * 20} {args.run}: {'주 실행' if args.run == MAIN_RUN else '보조 실행'} {'=' * 20}")
    print(f"케이스 {len(rows)}개 (PPP {int((labels == 0).sum())}, N 하나 {int((labels == 1).sum())}), "
          f"클래스 {len(set(objects))}개, 벡터 {tuple(hidden.shape[1:])}, 주 층 = 마지막 항목 {last}")
    print(f"바깥 {N_FOLDS}묶음의 클래스 수 {[sum(k == fold for k in folds.values()) for fold in range(N_FOLDS)]}, "
          f"이미지가 겹쳐 뺀 학습 케이스 {outer_dropped}")
    print(f"설정: probe L2 {L2}, 결합 L2 {COMBINER_L2}, 안쪽 묶음 {INNER_FOLDS}, 실질적 기준 ±{PRACTICAL_DELTA}")

    print("\n[클래스 내부 AUROC의 클래스 평균]  (모든 케이스 / 모델이 Yes라고 답한 케이스만)")
    for name in scores:
        print(f"  {name:<10}{mean_defined(by_class[name]):.3f}   {mean_defined(by_class_yes[name]):.3f}")

    print("\n[판정]  클래스마다의 차이의 평균 [95% 구간], 통계적 방향, 실질적 효과")
    main_line, main_record = verdict_line("주. 결합 - margin",
                                          paired_difference(by_class["combined"], by_class["margin"]))
    print(main_line)
    yes_line, yes_record = verdict_line("보조. 결합 - margin, 모델 Yes만",
                                        paired_difference(by_class_yes["combined"], by_class_yes["margin"]))
    print(yes_line)
    probe_line, probe_record = verdict_line("보조. probe 단독 - margin",
                                            paired_difference(by_class["probe"], by_class["margin"]))
    print(probe_line)
    print(f"  주 판정은 주 실행({MAIN_RUN})의 첫 줄이다. 보조 실행과 보조 줄은 같은 판정이 나오는지를 본다.")

    print("\n[묶음별]  결합 계수(표준화한 -margin, probe), 안쪽 점수와 바깥 점수의 분포(평균 ± 표준편차), Δ")
    fold_deltas = []
    for record in nested["folds"]:
        fold_classes = [name for name, fold in folds.items() if fold == record["fold"]]
        delta = statistics.mean(paired_difference({name: by_class["combined"][name] for name in fold_classes},
                                                  {name: by_class["margin"][name] for name in fold_classes}))
        fold_deltas.append(delta)
        weight = record["combiner"]["weight"]
        inner, outer = record["inner_scores"], record["outer_scores"]
        print(f"  묶음 {record['fold']}: 클래스 {len(fold_classes)}, 학습 {record['n_train']} (뺀 케이스 바깥 "
              f"{record['dropped_outer']}, 안쪽 {record['dropped_inner']}), 계수 {weight[0]:+.3f} {weight[1]:+.3f}, "
              f"안쪽 {inner['mean']:+.2f} ± {inner['std']:.2f}, 바깥 {outer['mean']:+.2f} ± {outer['std']:.2f}, "
              f"Δ {delta:+.3f}")

    shuffled = labels[torch.randperm(len(labels), generator=torch.Generator().manual_seed(CONTROL_SEED))]
    layer_auroc, control_auroc = [], []
    for layer in tqdm(range(hidden.shape[1]), desc="층별 probe 단독"):
        x = hidden[index, layer]
        probe_only = held_out_probe_scores(x, labels, rows, fold_of_case)[0]
        control = held_out_probe_scores(x, shuffled, rows, fold_of_case)[0]
        layer_auroc.append(mean_defined(within_class_auroc(probe_only, labels, objects)))
        control_auroc.append(mean_defined(within_class_auroc(control, shuffled, objects)))
    print(f"\n[층별 probe 단독의 클래스 내부 AUROC (판정 없음)]  margin 단독 {mean_defined(by_class['margin']):.3f}")
    print("  " + "  ".join(f"{layer}:{value:.3f}" for layer, value in enumerate(layer_auroc)))
    print("  control (레이블을 섞어 학습): "
          + "  ".join(f"{layer}:{value:.3f}" for layer, value in enumerate(control_auroc)))
    print(f"\n{LIMITS}")

    out_dir = PREDICTIONS_DIR / EXPERIMENT / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "source_run": args.run,
        "settings": {"l2": L2, "combiner_l2": COMBINER_L2, "n_folds": N_FOLDS, "inner_folds": INNER_FOLDS,
                     "practical_delta": PRACTICAL_DELTA, "control_seed": CONTROL_SEED, "layer": last},
        "git": git_state(),
        "folds": folds,
        "outer_dropped": outer_dropped,
        "case_ids": [row["case_id"] for row in rows],
        "labels": labels,
        "scores": scores,
        "by_class": by_class,
        "by_class_model_yes": by_class_yes,
        "nested_folds": nested["folds"],
        "fold_deltas": fold_deltas,
        "verdicts": {"main": main_record, "model_yes": yes_record, "probe_vs_margin": probe_record},
        "layer_auroc": layer_auroc,
        "control_auroc": control_auroc,
    }, out_dir / "recheck.pt")
    print(f"\n[ok] {out_dir / 'recheck.pt'}")


if __name__ == "__main__":
    main()
