"""
src/mivlm/analysis/probe.py 단위 테스트 (모델은 로딩하지 않는다)
pytest tests/unit/test_probe.py -v
"""
import json

import torch

from mivlm.analysis.pair_store import save_pair_record
from mivlm.analysis.probe import (
    accuracy_by_layer, class_folds, cross_validated_scores, fit_probe, load_pre_answer, presence_labels, probe_scores,
    slot_labels,
)


def _separable(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """첫 차원의 부호로 레이블이 정해지는 데이터: x는 (n, 6), y는 0 또는 1"""
    x = torch.randn(n, 6, generator=torch.Generator().manual_seed(seed))
    return x, (x[:, 0] > 0).long()


# ------------------------------------------------------------ load_pre_answer


def test_load_pre_answer_collects_one_vector_set_per_case_with_class_and_variant(tmp_path):
    cases = [{"case_id": f"pair_{pair}_{variant}", "object": name, "variant": variant}
             for pair, name in [("a", "cat"), ("b", "dog")] for variant in ["PPP", "PPN"]]
    with open(tmp_path / "results.jsonl", "w") as f:
        for pair in ["a", "b"]:
            record = {"pair": f"pair_{pair}", "comparisons": [], "cases": {
                variant: {"case_id": f"pair_{pair}_{variant}", "logit_margin": margin,
                          "pre_answer_hidden": torch.full((3, 4), float(value))}
                for variant, margin, value in [("PPP", 1.5, 1), ("PPN", -0.5, 2)]
            }}
            f.write(json.dumps({"pair": record["pair"], **save_pair_record(tmp_path, record)}) + "\n")

    rows, hidden = load_pre_answer(tmp_path, cases)

    assert rows == [
        {"case_id": "pair_a_PPP", "object": "cat", "variant": "PPP", "logit_margin": 1.5},
        {"case_id": "pair_a_PPN", "object": "cat", "variant": "PPN", "logit_margin": -0.5},
        {"case_id": "pair_b_PPP", "object": "dog", "variant": "PPP", "logit_margin": 1.5},
        {"case_id": "pair_b_PPN", "object": "dog", "variant": "PPN", "logit_margin": -0.5},
    ]
    assert hidden.shape == (4, 3, 4)  # 케이스 4개, 층 3개, 벡터 크기 4
    assert hidden[:, 0, 0].tolist() == [1.0, 2.0, 1.0, 2.0]


# ---------------------------------------------------------------- class_folds


def _cases(images_by_class: dict[str, list[int]]) -> list[dict]:
    return [{"object": name, "image_ids": image_ids} for name, image_ids in images_by_class.items()]


def test_classes_that_share_an_image_go_to_the_same_fold():
    # b와 c가 이미지 3을 같이 쓰고, c와 d가 이미지 5를 같이 쓴다. b, c, d는 한 덩어리다
    folds = class_folds(_cases({"a": [1, 2], "b": [3, 4], "c": [3, 5], "d": [5, 6], "e": [7], "f": [8]}), n_folds=3)

    assert folds["b"] == folds["c"] == folds["d"]
    assert set(folds) == {"a", "b", "c", "d", "e", "f"}


def test_no_image_appears_in_two_folds():
    cases = _cases({"a": [1, 2], "b": [3, 4], "c": [3, 5], "d": [5, 6], "e": [7], "f": [8]})
    folds = class_folds(cases, n_folds=3)

    folds_of_image = {}
    for case in cases:
        for image_id in case["image_ids"]:
            folds_of_image.setdefault(image_id, set()).add(folds[case["object"]])
    assert all(len(found) == 1 for found in folds_of_image.values())


def test_folds_are_filled_largest_group_first_into_the_smallest_fold():
    # 덩어리: {b, c, d} 3개, 그리고 a, e, f가 하나씩. 큰 것부터 넣으면 묶음 크기는 3, 2, 1이 된다
    folds = class_folds(_cases({"a": [1, 2], "b": [3, 4], "c": [3, 5], "d": [5, 6], "e": [7], "f": [8]}), n_folds=3)

    assert folds == {"b": 0, "c": 0, "d": 0, "a": 1, "e": 2, "f": 1}


# --------------------------------------------- presence_labels, slot_labels


def test_presence_uses_ppp_and_the_one_n_variants():
    rows = [{"variant": variant} for variant in ["PPP", "PPN", "PNN", "NNN", "NPP", "PNP"]]

    index, labels = presence_labels(rows)

    assert index.tolist() == [0, 1, 4, 5]  # PNN, NNN은 쓰지 않는다
    assert labels.tolist() == [0, 1, 1, 1]  # PPP만 0


def test_slot_label_is_where_the_n_is():
    rows = [{"variant": variant} for variant in ["PPP", "PPN", "PNN", "NNN", "NPP", "PNP"]]

    index, labels = slot_labels(rows)

    assert index.tolist() == [1, 4, 5]
    assert labels.tolist() == [2, 0, 1]  # PPN은 자리 3, NPP는 자리 1, PNP는 자리 2


# -------------------------------------------------- fit_probe, probe_scores


def test_probe_learns_a_linearly_separable_rule_and_applies_it_to_new_cases():
    x, y = _separable(400)

    probe = fit_probe(x[:300], y[:300], n_classes=2)
    scores = probe_scores(probe, x[300:])

    assert probe["weight"].shape == (2, 6)
    assert probe["bias"].shape == (2,)
    assert scores.shape == (100, 2)  # 케이스마다 클래스별 점수
    assert (scores.argmax(dim=1) == y[300:]).float().mean().item() > 0.9


def test_probe_with_identical_vectors_predicts_the_majority_class():
    # 층 0처럼 모든 케이스의 벡터가 같으면 정보가 없으므로, 학습 레이블에서 더 많은 쪽으로 찍는다
    x = torch.full((10, 4), 0.3)
    y = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])

    probe = fit_probe(x, y, n_classes=2)
    predictions = probe_scores(probe, torch.full((5, 4), 0.3)).argmax(dim=1)

    assert torch.isfinite(probe["weight"]).all()
    assert predictions.tolist() == [1] * 5


# ----------------------------------------------------- cross_validated_scores


def test_cross_validation_predicts_every_case_and_finds_the_rule():
    x, y = _separable(300)
    fold_of_case = torch.arange(300) % 5

    scores = cross_validated_scores(x, y, fold_of_case, n_classes=2)

    assert scores.shape == (300, 2)
    assert (scores.argmax(dim=1) == y).float().mean().item() > 0.9


def test_a_case_is_never_predicted_by_a_probe_trained_on_its_own_fold():
    # 묶음 0의 레이블을 뒤집어도 묶음 0의 점수는 그대로다: 그 점수는 묶음 0을 빼고 학습한 probe가 매긴 것이기 때문이다
    x, y = _separable(300)
    fold_of_case = torch.arange(300) % 5
    flipped = y.clone()
    flipped[fold_of_case == 0] = 1 - flipped[fold_of_case == 0]

    before = cross_validated_scores(x, y, fold_of_case, n_classes=2)
    after = cross_validated_scores(x, flipped, fold_of_case, n_classes=2)

    assert torch.equal(before[fold_of_case == 0], after[fold_of_case == 0])


def test_a_case_is_predicted_from_the_labels_of_the_other_folds():
    # 반대로, 묶음 0이 아닌 케이스의 레이블을 전부 뒤집으면 묶음 0의 예측은 뒤집힌 규칙을 따른다
    x, y = _separable(300)
    fold_of_case = torch.arange(300) % 5
    flipped = y.clone()
    flipped[fold_of_case != 0] = 1 - flipped[fold_of_case != 0]

    predictions = cross_validated_scores(x, flipped, fold_of_case, n_classes=2).argmax(dim=1)

    in_fold_0 = fold_of_case == 0
    assert (predictions[in_fold_0] == y[in_fold_0]).float().mean().item() < 0.1


# ---------------------------------------------------------- accuracy_by_layer


def test_accuracy_is_computed_per_layer_over_the_masked_cases_only():
    labels = torch.tensor([0, 1, 1, 0])
    predictions = torch.tensor([[0, 1, 1, 0],   # 층 0: 모두 맞음
                                [1, 1, 0, 0],   # 층 1: 케이스 0과 2가 틀림
                                [1, 0, 0, 1]])  # 층 2: 모두 틀림
    last_three = torch.tensor([False, True, True, True])

    assert accuracy_by_layer(predictions, labels, torch.ones(4, dtype=torch.bool)).tolist() == [1.0, 0.5, 0.0]
    assert torch.allclose(accuracy_by_layer(predictions, labels, last_three), torch.tensor([1.0, 2 / 3, 0.0]))

