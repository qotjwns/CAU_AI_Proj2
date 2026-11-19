"""
src/mivlm/evaluation/auroc.py 단위 테스트
pytest tests/unit/test_auroc.py -v
"""
import torch

from mivlm.evaluation.auroc import auroc


def test_auroc_is_1_when_separated_0_when_reversed_and_half_when_all_scores_are_equal():
    labels = torch.tensor([0, 0, 1, 1])
    scores = torch.tensor([[0.1, 0.2, 0.8, 0.9],   # 레이블 1의 점수가 모두 더 높다
                           [0.8, 0.9, 0.1, 0.2],   # 모두 더 낮다
                           [0.5, 0.5, 0.5, 0.5]])  # 모든 케이스의 점수가 같다 (층 0처럼)

    assert auroc(scores, labels).tolist() == [1.0, 0.0, 0.5]  # 앞의 차원(층)마다 하나씩


def test_auroc_counts_the_pairs_ranked_correctly_and_ties_as_half():
    # 레이블 1의 점수 [0.9, 0.4], 레이블 0의 점수 [0.4, 0.1, 0.7]. (레이블 1, 레이블 0) 쌍은 6개다
    #   0.9 대 0.4, 0.1, 0.7: 셋 다 옳은 순서
    #   0.4 대 0.4: 같음 / 0.4 대 0.1: 옳은 순서 / 0.4 대 0.7: 틀린 순서
    # 옳은 순서 4개, 같은 것 1개, 틀린 순서 1개
    labels = torch.tensor([1, 0, 1, 0, 0])
    scores = torch.tensor([[0.9, 0.4, 0.4, 0.1, 0.7]])

    assert torch.allclose(auroc(scores, labels), torch.tensor([(4 + 0.5) / 6]))


def test_auroc_of_scores_without_a_leading_dimension_is_a_single_number():
    labels = torch.tensor([1, 0, 1, 0, 0])
    scores = torch.tensor([0.9, 0.4, 0.4, 0.1, 0.7])  # 위 테스트와 같은 점수

    result = auroc(scores, labels)

    assert result.shape == ()
    assert torch.allclose(result, torch.tensor((4 + 0.5) / 6))


def test_auroc_does_not_depend_on_how_many_cases_each_label_has():
    # 레이블 0인 케이스를 그대로 세 번 넣어도 값은 같다 (정확도와 다른 점)
    labels = torch.tensor([1, 0, 1, 0, 0])
    scores = torch.tensor([[0.9, 0.4, 0.4, 0.1, 0.7]])
    more_labels = torch.cat([labels, torch.tensor([0, 0, 0, 0, 0, 0])])
    more_scores = torch.cat([scores, torch.tensor([[0.4, 0.1, 0.7, 0.4, 0.1, 0.7]])], dim=1)

    assert torch.allclose(auroc(more_scores, more_labels), auroc(scores, labels))
