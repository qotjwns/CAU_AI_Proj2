"""
src/mivlm/analysis/hidden_states.py 단위 테스트 (모델은 로딩하지 않는다)
pytest tests/unit/test_hidden_states.py -v
"""
from types import SimpleNamespace

import torch

from mivlm.analysis.hidden_states import extract_hidden_states

N_LAYERS, N_TOKENS, HIDDEN_SIZE, VOCAB_SIZE = 3, 6, 4, 5


class FakeModel:
    """실제 모델처럼 bf16 텐서를 돌려주는 가짜 모델

    hidden state: 층 i, 토큰 t의 값이 전부 i * 100 + t
    logit: 토큰 t, vocab v의 값이 t * 10 + v
    """
    device = torch.device("cpu")

    def __init__(self):
        self.weight = torch.ones(1, requires_grad=True)  # 실제 모델의 가중치처럼 gradient를 요구한다

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        tokens = torch.arange(kwargs["input_ids"].shape[1], dtype=torch.float32)
        hidden_states = tuple(
            ((layer * 100 + tokens).view(1, -1, 1).expand(1, -1, HIDDEN_SIZE) * self.weight).to(torch.bfloat16)
            for layer in range(N_LAYERS)
        )
        logits = ((tokens * 10).view(1, -1, 1) + torch.arange(VOCAB_SIZE)) * self.weight
        return SimpleNamespace(hidden_states=hidden_states, logits=logits.to(torch.bfloat16))


class FakeInputs(dict):
    """processor 결과처럼 .to(device)가 있는 dict"""

    def to(self, device):
        self.device = device
        return self


def _extract():
    model = FakeModel()
    inputs = FakeInputs(input_ids=torch.zeros(1, N_TOKENS, dtype=torch.long))
    return model, inputs, extract_hidden_states(model, inputs)


def test_shape_is_layers_tokens_hidden_size_without_batch():
    _, _, result = _extract()

    assert result["hidden_states"].shape == (N_LAYERS, N_TOKENS, HIDDEN_SIZE)


def test_layer_and_token_order_is_kept():
    _, _, result = _extract()
    hidden = result["hidden_states"]

    assert hidden[0, 0].tolist() == [0] * HIDDEN_SIZE
    assert hidden[1, 3].tolist() == [103] * HIDDEN_SIZE
    assert hidden[2, 5].tolist() == [205] * HIDDEN_SIZE


def test_bf16_becomes_float32_on_cpu():
    _, _, result = _extract()

    for name in ["hidden_states", "logits"]:
        assert result[name].dtype == torch.float32, name
        assert result[name].device.type == "cpu", name


def test_logits_are_from_the_last_token():
    _, _, result = _extract()

    # 마지막 토큰은 t = 5이므로 50 + v
    assert result["logits"].tolist() == [50, 51, 52, 53, 54]


def test_model_is_called_on_its_device_for_all_layers_without_gradient():
    model, inputs, result = _extract()

    assert inputs.device == model.device
    assert model.kwargs["input_ids"] is inputs["input_ids"]
    assert model.kwargs["output_hidden_states"] is True
    assert not result["hidden_states"].requires_grad
    assert not result["logits"].requires_grad
