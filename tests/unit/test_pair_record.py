"""
src/mivlm/analysis/pair_record.py 단위 테스트 (모델은 로딩하지 않는다)
pytest tests/unit/test_pair_record.py -v

가짜 pair: 글자 토큰 2개 → 이미지 3장(사이마다 글자 토큰 1개) → 글자 토큰 2개. 마지막 토큰이 답변 위치다.
이미지마다 토큰 수와, position id가 커지는 폭을 따로 정해서 "토큰 인덱스는 같은데 position id는 다른 경우"와
그 반대의 경우가 모두 나오게 했다.
"""
import pytest
import torch

from mivlm.analysis.hidden_states import compare_hidden_states
from mivlm.analysis.pair_record import COMPARISONS, build_pair_record

N_LAYERS, HIDDEN_SIZE = 3, 4
LAYOUT = {  # 변형마다 자리 1, 2, 3에 놓이는 이미지
    "PPP": ["P1", "P2", "P3"], "PPN": ["P1", "P2", "N1"], "PNP": ["P1", "N1", "P3"],
    "NPP": ["N1", "P2", "P3"], "PNN": ["P1", "N2", "N1"], "NNN": ["N3", "N2", "N1"],
}
IMAGE_IDS = {"P1": 11, "P2": 12, "P3": 13, "N1": 21, "N2": 22, "N3": 23}
N_TOKENS = {"P1": 2, "P2": 3, "P3": 2, "N1": 3, "N2": 3, "N3": 2}  # 이미지의 토큰 수
POSITION_WIDTH = {"P1": 2, "P2": 3, "P3": 2, "N1": 2, "N2": 2, "N3": 2}  # 이미지를 지나면 position id가 커지는 폭


def fake_observation(variant: str) -> dict:
    """LAYOUT대로 이미지를 놓았을 때의 observation. hidden state는 변형마다 다른 난수다"""
    token_spans, position_ids = [], [0, 1]  # 맨 앞의 글자 토큰 2개
    for name in LAYOUT[variant]:
        start, position_start = len(position_ids), position_ids[-1] + 1
        position_ids += [position_start + offset % POSITION_WIDTH[name] for offset in range(N_TOKENS[name])]
        token_spans.append((start, len(position_ids)))
        position_ids.append(position_start + POSITION_WIDTH[name])  # 이미지 뒤의 글자 토큰
    position_ids.append(position_ids[-1] + 1)  # 마지막 토큰 = 답변 위치

    generator = torch.Generator().manual_seed(list(LAYOUT).index(variant))
    return {
        "case_id": f"fake_pair_{variant}",
        "image_ids": [IMAGE_IDS[name] for name in LAYOUT[variant]],
        "token_spans": token_spans,
        "image_grid_thw": [[1, 2, 2 * N_TOKENS[name]] for name in LAYOUT[variant]],
        "position_ids": torch.tensor(position_ids).expand(3, -1).clone(),
        "hidden_states": torch.randn(N_LAYERS, len(position_ids), HIDDEN_SIZE, generator=generator),
        "yes_logit": 3.5,
        "no_logit": 1.25,
    }


@pytest.fixture
def observations():
    return {variant: fake_observation(variant) for variant in LAYOUT}


@pytest.fixture
def record(observations):
    return build_pair_record("fake_pair", observations)


# ---------------------------------------------------------------- COMPARISONS


def test_comparisons_are_the_12_agreed_ones_in_order():
    assert [(a_ref, b_ref) for _, _, a_ref, b_ref in COMPARISONS] == [
        (("NPP", 1), ("PNP", 2)), (("NPP", 1), ("PPN", 3)), (("PNP", 2), ("PPN", 3)),
        (("PPN", 3), ("PNN", 3)), (("PNN", 3), ("NNN", 3)), (("PPN", 3), ("NNN", 3)),
        (("PPP", 2), ("NPP", 2)), (("PNN", 2), ("NNN", 2)),
        (("PPP", 3), ("PNP", 3)), (("PPP", 3), ("NPP", 3)), (("PNP", 3), ("NPP", 3)),
        (("PPP", 2), ("PPN", 2)),
    ]
    assert [family for family, _, _, _ in COMPARISONS] == (
        ["slot_change"] * 3 + ["n1_at_slot_3"] * 3 + ["same_slot"] * 5 + ["same_prefix"]
    )


def test_every_comparison_pairs_the_same_image():
    for _, image, (a_variant, a_slot), (b_variant, b_slot) in COMPARISONS:
        assert LAYOUT[a_variant][a_slot - 1] == image
        assert LAYOUT[b_variant][b_slot - 1] == image


def test_comparison_families_match_what_changes():
    for family, _, (a_variant, a_slot), (b_variant, b_slot) in COMPARISONS:
        same_slot = a_slot == b_slot
        same_images_before = LAYOUT[a_variant][:a_slot - 1] == LAYOUT[b_variant][:b_slot - 1]

        assert same_slot == (family != "slot_change")
        assert same_images_before == (family == "same_prefix")
        if family == "n1_at_slot_3":
            assert (a_slot, b_slot) == (3, 3)


# ---------------------------------------------------------- build_pair_record


def test_record_has_six_cases_and_twelve_comparisons(record):
    assert record["pair"] == "fake_pair"
    assert list(record["cases"]) == list(LAYOUT)
    assert len(record["comparisons"]) == 12
    assert [comparison["family"] for comparison in record["comparisons"]] == [family for family, *_ in COMPARISONS]


def test_case_record_keeps_token_index_and_position_id_apart(record):
    # PNP: 글자 2 → P1(토큰 2개) → 글자 1 → N1(토큰 3개) → 글자 1 → P3(토큰 2개) → 글자 2
    # N1은 토큰이 3개지만 position id는 2만 커지므로, P3의 토큰 인덱스는 9이고 position id는 8이다
    case = record["cases"]["PNP"]

    assert case["case_id"] == "fake_pair_PNP"
    assert case["image_ids"] == [11, 21, 13]
    assert case["token_spans"] == [[2, 4], [5, 8], [9, 11]]
    assert case["position_starts"] == [2, 5, 8]
    assert case["image_grid_thw"] == [[1, 2, 4], [1, 2, 6], [1, 2, 4]]
    assert case["input_token_count"] == 13


def test_case_record_has_the_logits_and_their_margin(record):
    case = record["cases"]["PPP"]

    assert (case["yes_logit"], case["no_logit"]) == (3.5, 1.25)
    assert case["logit_margin"] == 2.25


def test_pre_answer_hidden_is_the_last_token_of_every_layer(observations, record):
    for variant, observation in observations.items():
        pre_answer_hidden = record["cases"][variant]["pre_answer_hidden"]

        assert pre_answer_hidden.shape == (N_LAYERS, HIDDEN_SIZE)
        assert pre_answer_hidden.dtype == torch.float32
        assert torch.equal(pre_answer_hidden, observation["hidden_states"][:, -1])


def test_pre_answer_hidden_does_not_hold_on_to_all_hidden_states(record):
    # 전체 hidden state의 일부를 가리키는 텐서를 그대로 저장하면 전체가 파일에 들어간다
    pre_answer_hidden = record["cases"]["PPP"]["pre_answer_hidden"]

    assert pre_answer_hidden.untyped_storage().nbytes() == N_LAYERS * HIDDEN_SIZE * 4  # float32는 값 하나에 4바이트


def test_comparison_metrics_are_computed_on_the_image_tokens(observations, record):
    # 7번째 비교: P2가 PPP의 자리 2(토큰 5~7)와 NPP의 자리 2(토큰 6~8)에 있다
    comparison = record["comparisons"][6]
    expected = compare_hidden_states(observations["PPP"]["hidden_states"][:, 5:8],
                                     observations["NPP"]["hidden_states"][:, 6:9])

    assert (comparison["image"], comparison["image_id"]) == ("P2", 12)
    assert comparison["token_cosine"].shape == (N_LAYERS, 3)
    assert comparison["cosine_of_means"].shape == (N_LAYERS,)
    assert comparison["relative_l2"].shape == (N_LAYERS,)
    for name, values in expected.items():
        assert torch.equal(comparison[name], values), name


def test_comparison_sides_record_token_span_and_position_start(record):
    # P2: PPP에서는 P1(토큰 2개) 뒤, NPP에서는 N1(토큰 3개) 뒤. 토큰 인덱스는 다르지만 position id는 둘 다 5에서 시작한다
    comparison = record["comparisons"][6]

    assert comparison["a"] == {"variant": "PPP", "slot": 2, "token_span": [5, 8], "position_start": 5}
    assert comparison["b"] == {"variant": "NPP", "slot": 2, "token_span": [6, 9], "position_start": 5}


def test_same_token_index_and_same_position_ids_are_separate_conditions(record):
    # (same_token_index, same_position_ids)를 COMPARISONS 순서대로. 네 가지 조합이 모두 나온다
    assert [(c["same_token_index"], c["same_position_ids"]) for c in record["comparisons"]] == [
        (False, False), (False, False), (False, False),  # slot_change: 자리가 다르므로 둘 다 다르다
        (True, False),   # N1 자리 3: PPN vs PNN. P2 → N2는 토큰 수는 같고 position 폭이 다르다
        (True, True),    # N1 자리 3: PNN vs NNN. P1 → N3은 토큰 수도 position 폭도 같다
        (True, False),   # N1 자리 3: PPN vs NNN
        (False, True),   # P2 자리 2: PPP vs NPP. P1 → N1은 토큰 수는 다르고 position 폭이 같다
        (True, True),    # N2 자리 2: PNN vs NNN
        (True, False),   # P3 자리 3: PPP vs PNP
        (False, True),   # P3 자리 3: PPP vs NPP
        (False, False),  # P3 자리 3: PNP vs NPP
        (True, True),    # same_prefix: 앞부분이 완전히 같다
    ]


def test_same_position_ids_compares_all_three_axes(observations):
    # 세로 축(1번) 한 칸만 달라도 같지 않다고 본다
    start, _ = observations["PPN"]["token_spans"][1]
    observations["PPN"]["position_ids"][1, start] += 1

    record = build_pair_record("fake_pair", observations)

    same_prefix = record["comparisons"][11]  # P2: PPP 자리 2 vs PPN 자리 2
    assert same_prefix["same_token_index"]
    assert not same_prefix["same_position_ids"]


def test_mean_vectors_are_not_stored(record):
    for comparison in record["comparisons"]:
        assert set(comparison) == {
            "family", "image", "image_id", "a", "b", "same_token_index", "same_position_ids",
            "token_cosine", "cosine_of_means", "relative_l2",
        }


def test_different_images_are_rejected(observations):
    observations["NPP"]["image_ids"][0] = 99  # NPP의 자리 1이 N1이 아니게 된다

    with pytest.raises(ValueError, match="같은 이미지가 아니다"):
        build_pair_record("fake_pair", observations)


def test_different_token_counts_are_rejected(observations):
    start, end = observations["NPP"]["token_spans"][0]
    observations["NPP"]["token_spans"][0] = (start, end - 1)  # 같은 N1인데 토큰이 하나 적다

    with pytest.raises(ValueError, match="토큰 수가 다르다"):
        build_pair_record("fake_pair", observations)


def test_missing_variant_is_an_error(observations):
    del observations["NNN"]

    with pytest.raises(KeyError):
        build_pair_record("fake_pair", observations)
