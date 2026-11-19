"""
src/mivlm/analysis/evidence_link.py 단위 테스트 (모델도, 저장된 결과도 쓰지 않는다)
pytest tests/unit/test_evidence_link.py -v
"""
import pytest

from mivlm.analysis.evidence_link import (
    DROP_MODELS, LEVEL_MODELS, class_mean, drop_observations, level_observations, n_strength_bins,
    out_of_fold_predictions, practical_verdict, with_single_margins,
)


def _pair(single_ppn: list[float] = (3.0, 1.0, -4.0)) -> dict:
    """cat 클래스의 pair 하나. P의 단일 margin은 3.0, 1.0, 2.0이고 N(이미지 9)의 단일 margin은 -4.0이다"""
    def case(variant, margin, single, image_ids):
        return {"object": "cat", "variant": variant, "logit_margin": margin, "single_margins": list(single),
                "image_ids": image_ids, "prediction": "Yes" if margin > 0 else "No"}
    return {"pair": "group_01_cat_1", "cases": {
        "PPP": case("PPP", 2.0, [3.0, 1.0, 2.0], [1, 2, 3]),
        "PPN": case("PPN", -1.0, single_ppn, [1, 2, 9]),
        "PNP": case("PNP", 0.5, [3.0, -4.0, 2.0], [1, 9, 3]),
        "NPP": case("NPP", 1.5, [-4.0, 1.0, 2.0], [9, 2, 3]),
    }}


def _linear_observations() -> list[dict]:
    """클래스 a, b, c에 관측 둘씩 (PPN, PNP). target = 2 + 3 x min으로 정확히 정해진다"""
    slots = [("a", "PPN"), ("a", "PNP"), ("b", "PPN"), ("b", "PNP"), ("c", "PPN"), ("c", "PNP")]
    return [{"object": name, "slot": slot, "min": float(x), "target": 2.0 + 3.0 * x}
            for x, (name, slot) in enumerate(slots)]


MODELS = {"intercept": (None, False), "slot": (None, True), "min": ("min", False), "min+slot": ("min", True)}
ONE_CLASS = {"a": "a", "b": "b", "c": "c"}


# --------------------------------------------------------- with_single_margins


def test_single_margins_are_attached_to_the_joint_rows_case_by_case():
    a_rows = [{"case_id": "x_PPP", "logit_margin": 2.0}, {"case_id": "x_PPN", "logit_margin": -1.0}]
    b_rows = [{"case_id": "x_PPP", "logit_margin": 1.0, "image_margins": [3.0, 1.0, 2.0]},
              {"case_id": "x_PPN", "logit_margin": -4.0, "image_margins": [3.0, 1.0, -4.0]}]

    rows = with_single_margins(a_rows, b_rows)

    assert rows == [{"case_id": "x_PPP", "logit_margin": 2.0, "single_margins": [3.0, 1.0, 2.0]},
                    {"case_id": "x_PPN", "logit_margin": -1.0, "single_margins": [3.0, 1.0, -4.0]}]  # margin은 A의 것


def test_rows_in_a_different_case_order_are_an_error():
    a_rows = [{"case_id": "x_PPP", "logit_margin": 2.0}, {"case_id": "x_PPN", "logit_margin": -1.0}]

    b_rows = [{"case_id": "x_PPN", "image_margins": []}, {"case_id": "x_PPP", "image_margins": []}]

    with pytest.raises(ValueError):
        with_single_margins(a_rows, b_rows)


# ------------------------------------------- drop_observations, level_observations


def test_drop_observations_are_hand_computed_for_each_variant():
    by_slot = {observation["slot"]: observation for observation in drop_observations([_pair()])}

    assert list(by_slot) == ["PPN", "PNP", "NPP"]
    # PPN: 공동 margin 2.0 -> -1.0. 단일 margin [3, 1, 2] -> [3, 1, -4]
    assert by_slot["PPN"] == {"object": "cat", "slot": "PPN", "target": 3.0, "min": 5.0, "mean": 2.0, "n": 4.0,
                              "n_image": 9, "n_single": -4.0, "n_is_lowest": True, "false_yes": False}
    # PNP: 공동 margin 2.0 -> 0.5 (Yes로 답함). 단일 margin [3, 1, 2] -> [3, -4, 2]
    assert by_slot["PNP"]["target"] == 1.5
    assert by_slot["PNP"]["min"] == 5.0  # min은 1에서 -4로
    assert by_slot["PNP"]["mean"] == pytest.approx(5 / 3)  # (바뀐 P의 1.0 - N의 -4.0) / 3
    assert by_slot["PNP"]["false_yes"] is True
    # NPP: 공동 margin 2.0 -> 1.5. 바뀐 P의 단일 margin은 3.0
    assert by_slot["NPP"]["target"] == 0.5
    assert by_slot["NPP"]["mean"] == pytest.approx(7 / 3)


def test_n_is_lowest_is_false_when_a_p_image_has_a_lower_single_margin_than_the_n_image():
    observation = drop_observations([_pair(single_ppn=(3.0, -6.0, -4.0))])[0]  # 자리 2의 P가 -6.0이다

    assert observation["n_is_lowest"] is False
    assert observation["n"] == 4.0  # n은 N 이미지의 값만 본다


def test_level_observations_have_one_row_per_case_with_the_joint_margin_as_target():
    observations = level_observations([_pair()])

    assert [observation["slot"] for observation in observations] == ["PPP", "PPN", "PNP", "NPP"]
    assert observations[0] == {"object": "cat", "slot": "PPP", "target": 2.0, "min": 1.0, "mean": 2.0}
    assert observations[1] == {"object": "cat", "slot": "PPN", "target": -1.0, "min": -4.0, "mean": 0.0}


def test_level_models_are_the_drop_models_without_the_n_only_ones():
    assert set(DROP_MODELS) == {"intercept", "slot", "n", "min", "mean", "n+slot", "min+slot", "mean+slot"}
    assert set(LEVEL_MODELS) == {"intercept", "slot", "min", "mean", "min+slot", "mean+slot"}


# ------------------------------------------------------ out_of_fold_predictions


def test_each_class_is_predicted_from_the_other_classes():
    # target: a는 2, 5 / b는 8, 11 / c는 14, 17
    predictions = out_of_fold_predictions(_linear_observations(), ONE_CLASS, MODELS)

    # 절편만: 나머지 두 클래스의 target 평균. a는 (8 + 11 + 14 + 17) / 4, b는 (2 + 5 + 14 + 17) / 4, c는 (2 + 5 + 8 + 11) / 4
    assert predictions["intercept"] == pytest.approx([12.5, 12.5, 9.5, 9.5, 6.5, 6.5])
    # 자리마다 절편: 나머지 두 클래스에서 같은 자리의 target 평균. a의 PPN은 (8 + 14) / 2, PNP는 (11 + 17) / 2
    assert predictions["slot"] == pytest.approx([11.0, 14.0, 8.0, 11.0, 5.0, 8.0])
    # 설명 변수가 있으면 나머지 클래스에서 기울기 3, 절편 2를 정확히 찾아내므로 뺀 클래스도 정확히 맞힌다
    assert predictions["min"] == pytest.approx([2.0, 5.0, 8.0, 11.0, 14.0, 17.0])
    assert predictions["min+slot"] == pytest.approx([2.0, 5.0, 8.0, 11.0, 14.0, 17.0])


def test_the_targets_of_the_held_out_class_do_not_affect_its_own_predictions():
    observations = _linear_observations()
    changed = [{**observation, "target": 100.0} if observation["object"] == "a" else observation
               for observation in observations]

    before = out_of_fold_predictions(observations, ONE_CLASS, MODELS)
    after = out_of_fold_predictions(changed, ONE_CLASS, MODELS)

    for model in MODELS:
        assert after[model][:2] == pytest.approx(before[model][:2])  # a의 예측은 b, c만으로 만든다
    assert after["intercept"][2:] != pytest.approx(before["intercept"][2:])  # b, c의 예측은 a의 target을 쓴다


def test_classes_in_the_same_fold_are_held_out_together():
    # a와 b를 같이 빼면 둘의 예측은 c만으로 만든다: (14 + 17) / 2. c의 예측은 a, b로 만든다: (2 + 5 + 8 + 11) / 4
    together = {"a": 0, "b": 0, "c": 1}
    predictions = out_of_fold_predictions(_linear_observations(), together, {"intercept": (None, False)})

    assert predictions["intercept"] == pytest.approx([15.5, 15.5, 15.5, 15.5, 6.5, 6.5])


# ------------------------------------------------- class_mean, n_strength_bins


def test_class_mean_averages_the_values_of_each_class_in_first_seen_order():
    observations = [{"object": "b"}, {"object": "a"}, {"object": "b"}, {"object": "a"}, {"object": "a"}]

    assert class_mean(observations, [1.0, 10.0, 3.0, 20.0, 30.0]) == {"b": 2.0, "a": 20.0}


def test_n_strength_bins_split_the_n_images_of_a_class_into_thirds_by_single_margin():
    cat = [("cat", image, single) for image, single in [(12, -3.0), (11, -5.0), (13, -1.0), (11, -5.0)]]
    dog = [("dog", image, float(image - 30)) for image in range(21, 29)]  # 이미지 21 ~ 28, 단일 margin -9 ~ -2
    observations = [{"object": name, "n_image": image, "n_single": single} for name, image, single in cat + dog]

    bins = n_strength_bins(observations)

    assert bins[:4] == [1, 0, 2, 0]  # cat: -5.0이 가장 강하게 No(0), -3.0이 가운데(1), -1.0이 가장 약하게 No(2)
    assert bins[4:] == [0, 0, 0, 1, 1, 1, 2, 2]  # 8장은 3, 3, 2장으로 나뉜다


# ------------------------------------------------------------ practical_verdict


def test_practical_verdict_needs_the_whole_interval_beyond_delta():
    assert practical_verdict(0.06, 0.20, 0.05) == "앞 모델이 실질적으로 우월"
    assert practical_verdict(-0.20, -0.06, 0.05) == "뒤 모델이 실질적으로 우월"
    assert practical_verdict(-0.03, 0.04, 0.05) == "실질적 차이 없음"
    assert practical_verdict(0.01, 0.04, 0.05) == "실질적 차이 없음"  # 방향은 있어도 delta 안이면 차이 없음이다
    assert practical_verdict(0.01, 0.20, 0.05) == "미결정"  # 0은 제외하지만 delta를 다 넘지 못했다
    assert practical_verdict(0.05, 0.20, 0.05) == "미결정"  # 하한이 delta와 같으면 넘은 것이 아니다
